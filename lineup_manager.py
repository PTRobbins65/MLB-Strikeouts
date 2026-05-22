"""
MLB Strikeout Pipeline — Lineup Manager
Handles the two-phase lineup lifecycle:

  Phase 1 — PROJECTED  (24 hrs before game)
      Build an expected batting order from each team's recent lineup history.
      Uses the last N confirmed lineups to estimate who will start and where.

  Phase 2 — CONFIRMED  (2–4 hrs before game, once StatsAPI publishes)
      Replace the projection with the real batting order from the official
      MLB StatsAPI.  A background polling loop watches for the transition.

Public interface
----------------
  manager = LineupManager()
  games   = manager.get_today_games()          # list of game dicts
  lineup  = manager.get_lineup(game_pk)        # ProjectedLineup or ConfirmedLineup
  manager.start_polling()                      # background thread, updates cache
  manager.stop_polling()
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import requests

from config import (
    MLB_API_BASE,
    SCHEDULE_HYDRATE,
    CONFIRMED_STATUS_CODES,
    LINEUP_POLL_INTERVAL_SECONDS,
    RAW_DIR,
)

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BatterSlot:
    batting_order: int          # 1–9
    player_id: int
    full_name: str
    position: str
    bats: str = "U"             # L / R / S / U (unknown until roster lookup)
    # Feature placeholders — populated by FeatureBuilder
    k_pct: Optional[float] = None
    o_swing_pct: Optional[float] = None
    contact_pct: Optional[float] = None
    swstr_pct: Optional[float] = None
    wrc_plus: Optional[float] = None


@dataclass
class LineupCard:
    game_pk: int
    game_date: str
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    probable_pitcher_home: Optional[Dict] = None   # {id, fullName, throws}
    probable_pitcher_away: Optional[Dict] = None
    home_batters: List[BatterSlot] = field(default_factory=list)
    away_batters: List[BatterSlot] = field(default_factory=list)
    confirmed: bool = False                        # True once StatsAPI has real lineup
    last_updated: Optional[datetime] = None
    game_status: str = "Scheduled"


# ── Helper: call the official MLB StatsAPI ────────────────────────────────────

def _mlb_get(endpoint: str, params: dict = None) -> dict:
    """Thin wrapper around the MLB StatsAPI. Raises on HTTP errors."""
    url = f"{MLB_API_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── Lineup Manager ─────────────────────────────────────────────────────────────

class LineupManager:
    """
    Manages projected and confirmed lineup data for a given game date.
    Thread-safe: a background polling loop updates the internal cache.
    """

    def __init__(self, target_date: Optional[date] = None):
        self.target_date = target_date or date.today()
        self._cache: Dict[int, LineupCard] = {}              # game_pk -> LineupCard
        self._projected_cache: Dict[int, List[BatterSlot]] = {}  # team_id -> batters
        self._lock = threading.Lock()
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────

    def get_today_games(self) -> List[Dict]:
        """
        Return a list of today's scheduled games from the official schedule
        endpoint, including probable pitchers.

        Each dict has keys: game_pk, away_team, home_team, game_time (ET),
                            probable_pitcher_away, probable_pitcher_home,
                            game_status.
        """
        date_str = self.target_date.strftime("%Y-%m-%d")
        logger.info(f"Fetching schedule for {date_str}")

        # NOTE: no "fields" filter here — it conflicts with hydrate and strips
        # nested sub-fields like pitchHand.code from probablePitcher.
        data = _mlb_get("schedule", params={
            "sportId": 1,
            "date": date_str,
            "hydrate": SCHEDULE_HYDRATE,
        })

        games = []
        for date_entry in data.get("dates", []):
            for g in date_entry.get("games", []):
                away = g["teams"]["away"]
                home = g["teams"]["home"]

                venue   = g.get("venue", {})
                game_info = {
                    "game_pk":   g["gamePk"],
                    "game_date": date_str,
                    "away_team": away["team"]["name"],
                    "away_team_id": away["team"]["id"],
                    "home_team": home["team"]["name"],
                    "home_team_id": home["team"]["id"],
                    "game_status": g["status"]["detailedState"],
                    "game_time": g.get("gameDate", ""),
                    "venue_id": venue.get("id", 680),
                    "probable_pitcher_away": self._extract_probable(away),
                    "probable_pitcher_home": self._extract_probable(home),
                }
                games.append(game_info)

                # Seed cache with a stub LineupCard
                with self._lock:
                    if g["gamePk"] not in self._cache:
                        self._cache[g["gamePk"]] = LineupCard(
                            game_pk=g["gamePk"],
                            game_date=date_str,
                            home_team=home["team"]["name"],
                            away_team=away["team"]["name"],
                            home_team_id=home["team"]["id"],
                            away_team_id=away["team"]["id"],
                            probable_pitcher_home=game_info["probable_pitcher_home"],
                            probable_pitcher_away=game_info["probable_pitcher_away"],
                            game_status=g["status"]["detailedState"],
                        )

        pitchers_found = sum(
            1 for g in games
            if g.get("probable_pitcher_home") or g.get("probable_pitcher_away")
        )
        logger.info(
            f"Found {len(games)} games on {date_str} "
            f"({pitchers_found} games with at least one probable pitcher)"
        )
        if len(games) > 0 and pitchers_found == 0:
            logger.warning(
                "No probable pitchers returned by StatsAPI for any game — "
                "predictions will be empty. Check hydrate and schedule endpoint."
            )
        return games

    def get_lineup(self, game_pk: int) -> Optional[LineupCard]:
        """
        Return the best available lineup for a game.
        If the confirmed lineup is published, it will be used.
        Otherwise returns the projected lineup (may have empty batter lists).
        """
        with self._lock:
            card = self._cache.get(game_pk)

        if card is None:
            logger.warning(f"game_pk {game_pk} not in cache — call get_today_games() first")
            return None

        # If not yet confirmed, attempt a live fetch before returning
        if not card.confirmed:
            self._refresh_lineup(game_pk)
            with self._lock:
                card = self._cache.get(game_pk)

        return card

    def build_projected_lineup(
        self,
        team_id: int,
        game_pk: int,
        pitcher_throws: str,
        n_recent: int = 10,
    ) -> List[BatterSlot]:
        """
        Build a projected 9-batter lineup from a team's recent boxscore history.

        Uses the boxscore endpoint (not schedule hydrate=lineups, which doesn't
        return completed-game batting orders).  Results are cached per team_id
        so each team is only fetched once per pipeline run even if they appear
        in multiple games (doubleheaders).

        Parameters
        ----------
        team_id        : MLB team ID
        game_pk        : today's game_pk (excluded from history)
        pitcher_throws : "R" or "L"
        n_recent       : number of recent completed games to sample
        """
        # Return cached result if we already built this team's projection today
        if team_id in self._projected_cache:
            logger.debug(f"Projected lineup cache hit for team_id={team_id}")
            return self._projected_cache[team_id]

        logger.info(f"Building projected lineup for team_id={team_id} vs {pitcher_throws}HP")

        # Step 1: get recent game_pks for this team via schedule (no hydrate needed)
        end_date   = self.target_date.strftime("%Y-%m-%d")
        start_date = (self.target_date - timedelta(days=30)).strftime("%Y-%m-%d")

        try:
            sched = _mlb_get("schedule", params={
                "sportId":   1,
                "teamId":    team_id,
                "startDate": start_date,
                "endDate":   end_date,
                "gameType":  "R",
            })
        except Exception as exc:
            logger.warning(f"Schedule fetch failed for team_id={team_id}: {exc}")
            self._projected_cache[team_id] = []
            return []

        # Collect completed game_pks (most recent first)
        recent_pks: List[tuple] = []   # (game_pk, home_team_id)
        for date_entry in reversed(sched.get("dates", [])):
            for g in reversed(date_entry.get("games", [])):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                if g["gamePk"] == game_pk:
                    continue
                home_id = g["teams"]["home"]["team"]["id"]
                recent_pks.append((g["gamePk"], home_id))
                if len(recent_pks) >= n_recent:
                    break
            if len(recent_pks) >= n_recent:
                break

        if not recent_pks:
            logger.warning(f"No recent completed games found for team_id={team_id}")
            self._projected_cache[team_id] = []
            return []

        # Step 2: fetch boxscore for each game and extract batting order
        slot_votes: Dict[int, Dict[int, int]] = {}   # player_id -> {slot: count}
        slot_names: Dict[int, str] = {}
        slot_pos:   Dict[int, str] = {}
        games_counted = 0

        for pk, home_id in recent_pks:
            try:
                boxscore = _mlb_get(f"game/{pk}/boxscore")
            except Exception as exc:
                logger.debug(f"Boxscore fetch failed for game_pk={pk}: {exc}")
                continue

            side      = "home" if home_id == team_id else "away"
            team_data = boxscore.get("teams", {}).get(side, {})
            batters   = self._parse_boxscore_lineup(team_data)

            if not batters:
                continue

            games_counted += 1
            for b in batters:
                slot_votes.setdefault(b.player_id, {})
                slot_votes[b.player_id][b.batting_order] = (
                    slot_votes[b.player_id].get(b.batting_order, 0) + 1
                )
                slot_names[b.player_id] = b.full_name
                slot_pos[b.player_id]   = b.position

        if not slot_votes:
            logger.warning(f"No batting order data found for team_id={team_id} "
                           f"across {len(recent_pks)} games checked")
            self._projected_cache[team_id] = []
            return []

        # Step 3: for each slot 1–9 pick the player with the most appearances
        slot_to_player: Dict[int, BatterSlot] = {}
        for pid, votes in slot_votes.items():
            best_slot = max(votes, key=votes.get)
            count     = votes[best_slot]
            existing  = slot_to_player.get(best_slot)
            if existing is None or count > (existing.k_pct or 0):
                slot_to_player[best_slot] = BatterSlot(
                    batting_order = best_slot,
                    player_id     = pid,
                    full_name     = slot_names[pid],
                    position      = slot_pos[pid],
                    k_pct         = count,   # vote count; overwritten by FeatureBuilder
                )

        projected = sorted(slot_to_player.values(), key=lambda b: b.batting_order)
        logger.info(
            f"Projected {len(projected)} batters for team_id={team_id} "
            f"from {games_counted} recent games"
        )

        self._projected_cache[team_id] = projected
        return projected

    # ── Polling loop ──────────────────────────────────────────────────────

    def start_polling(self):
        """
        Start a background thread that re-checks all cached games for
        confirmed lineups at LINEUP_POLL_INTERVAL_SECONDS intervals.
        Call stop_polling() to shut it down.
        """
        if self._poll_thread and self._poll_thread.is_alive():
            logger.warning("Polling thread already running")
            return

        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="lineup-poller",
            daemon=True,
        )
        self._poll_thread.start()
        logger.info(f"Lineup polling started (every {LINEUP_POLL_INTERVAL_SECONDS}s)")

    def stop_polling(self):
        """Stop the background polling thread."""
        self._stop_event.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=10)
        logger.info("Lineup polling stopped")

    def _poll_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                game_pks = list(self._cache.keys())

            pending = [
                pk for pk in game_pks
                if not self._cache[pk].confirmed
                and self._cache[pk].game_status not in ("Final", "Game Over")
            ]

            if pending:
                logger.debug(f"Polling {len(pending)} games for lineup updates")
                for pk in pending:
                    try:
                        self._refresh_lineup(pk)
                    except Exception as e:
                        logger.error(f"Error refreshing lineup for game_pk={pk}: {e}")
            else:
                logger.debug("All lineups confirmed — polling idle")

            self._stop_event.wait(LINEUP_POLL_INTERVAL_SECONDS)

    def _refresh_lineup(self, game_pk: int):
        """
        Fetch live game data from StatsAPI and update the LineupCard cache.
        If the lineup has been confirmed, populates home_batters / away_batters.
        """
        try:
            data = _mlb_get(f"game/{game_pk}/boxscore")
        except Exception as e:
            logger.error(f"StatsAPI boxscore error for game_pk={game_pk}: {e}")
            return

        teams = data.get("teams", {})
        home_batters = self._parse_boxscore_lineup(teams.get("home", {}))
        away_batters = self._parse_boxscore_lineup(teams.get("away", {}))

        # The lineup is considered "confirmed" once at least 7 batters appear
        confirmed = len(home_batters) >= 7 and len(away_batters) >= 7

        # Also pull live game status
        try:
            linescore = _mlb_get(f"game/{game_pk}/linescore")
            status = linescore.get("inningState", "")
        except Exception:
            status = ""

        with self._lock:
            card = self._cache.get(game_pk)
            if card is None:
                return

            if confirmed and not card.confirmed:
                logger.info(
                    f"✓ Confirmed lineup for game_pk={game_pk} "
                    f"({card.away_team} @ {card.home_team})"
                )

            card.home_batters  = home_batters if home_batters else card.home_batters
            card.away_batters  = away_batters if away_batters else card.away_batters
            card.confirmed     = confirmed
            card.last_updated  = datetime.now()
            if status:
                card.game_status = status

    # ── Parsing helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_probable(team_dict: dict) -> Optional[Dict]:
        pp = team_dict.get("probablePitcher")
        if not pp:
            team_name = team_dict.get("team", {}).get("name", "?")
            logger.debug(f"No probablePitcher key for {team_name} — "
                         f"team_dict keys: {list(team_dict.keys())}")
            return None
        pitcher = {
            "id":       pp.get("id"),
            "fullName": pp.get("fullName"),
            "throws":   pp.get("pitchHand", {}).get("code", "R"),
        }
        logger.debug(f"Probable pitcher found: {pitcher['fullName']} "
                     f"(id={pitcher['id']}, throws={pitcher['throws']})")
        return pitcher

    @staticmethod
    def _parse_boxscore_lineup(team_data: dict) -> List[BatterSlot]:
        """
        Extract ordered batter list from a boxscore 'team' dict.
        Returns slots sorted by battingOrder (1–9).
        """
        players = team_data.get("players", {})
        batting_order_raw = team_data.get("battingOrder", [])

        # battingOrder is a list of player IDs in slot order
        # but boxscore also carries battingOrder per player
        batters = []
        for pid_str, player_data in players.items():
            bo = player_data.get("battingOrder")
            if bo is None:
                continue
            try:
                bo_int = int(str(bo).rstrip("0") or "0")   # "100" -> 1, "200" -> 2, etc.
            except ValueError:
                continue
            if bo_int < 1 or bo_int > 9:
                continue

            person   = player_data.get("person", {})
            pos_data = player_data.get("position", {})

            batters.append(BatterSlot(
                batting_order = bo_int,
                player_id     = person.get("id", 0),
                full_name     = person.get("fullName", "Unknown"),
                position      = pos_data.get("abbreviation", "?"),
            ))

        return sorted(batters, key=lambda b: b.batting_order)

    # ── Serialisation helpers ─────────────────────────────────────────────

    def save_lineup_snapshot(self, game_pk: int):
        """Persist current LineupCard to disk as JSON (for audit / replay)."""
        with self._lock:
            card = self._cache.get(game_pk)
        if card is None:
            return

        out_path = RAW_DIR / f"lineup_{game_pk}_{date.today()}.json"
        payload = {
            "game_pk":    card.game_pk,
            "game_date":  card.game_date,
            "home_team":  card.home_team,
            "away_team":  card.away_team,
            "confirmed":  card.confirmed,
            "last_updated": card.last_updated.isoformat() if card.last_updated else None,
            "probable_pitcher_home": card.probable_pitcher_home,
            "probable_pitcher_away": card.probable_pitcher_away,
            "home_batters": [vars(b) for b in card.home_batters],
            "away_batters": [vars(b) for b in card.away_batters],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        logger.info(f"Lineup snapshot saved -> {out_path.name}")
