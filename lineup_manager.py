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
        self._cache: Dict[int, LineupCard] = {}       # game_pk -> LineupCard
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

        data = _mlb_get("schedule", params={
            "sportId": 1,
            "date": date_str,
            "hydrate": SCHEDULE_HYDRATE,
            "fields": (
                "dates,games,gamePk,status,abstractGameState,detailedState,"
                "teams,away,home,team,id,name,abbreviation,"
                "probablePitcher,id,fullName,pitchHand,"
                "lineups,awayPlayers,homePlayers,"
                "battingOrder,playerName,primaryPosition,"
                "gameDate"
            ),
        })

        games = []
        for date_entry in data.get("dates", []):
            for g in date_entry.get("games", []):
                away = g["teams"]["away"]
                home = g["teams"]["home"]

                game_info = {
                    "game_pk":   g["gamePk"],
                    "game_date": date_str,
                    "away_team": away["team"]["name"],
                    "away_team_id": away["team"]["id"],
                    "home_team": home["team"]["name"],
                    "home_team_id": home["team"]["id"],
                    "game_status": g["status"]["detailedState"],
                    "game_time": g.get("gameDate", ""),
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

        logger.info(f"Found {len(games)} games on {date_str}")
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
        n_recent: int = 15,
    ) -> List[BatterSlot]:
        """
        Build a projected lineup from a team's recent lineup history.
        Queries the StatsAPI for the last n_recent completed games and
        uses the most-frequent batting-order slots to estimate starters.

        Parameters
        ----------
        team_id         : MLB team ID
        game_pk         : target game (used to filter out future games)
        pitcher_throws  : "R" or "L" — we match platoon splits
        n_recent        : number of recent games to sample
        """
        logger.info(f"Building projected lineup for team_id={team_id} vs {pitcher_throws}HP")

        # Get last n_recent completed games for this team
        end_date   = self.target_date.strftime("%Y-%m-%d")
        start_date = (self.target_date - timedelta(days=45)).strftime("%Y-%m-%d")

        data = _mlb_get("schedule", params={
            "sportId": 1,
            "teamId": team_id,
            "startDate": start_date,
            "endDate": end_date,
            "hydrate": "lineups,probablePitcher(pitchHand),team",
            "gameType": "R",           # regular season only
        })

        # Collect batting orders from completed games
        slot_votes: Dict[int, Dict[int, int]] = {}  # player_id -> {slot: count}
        slot_names: Dict[int, str] = {}
        slot_pos:   Dict[int, str] = {}
        games_counted = 0

        for date_entry in reversed(data.get("dates", [])):
            if games_counted >= n_recent:
                break
            for g in reversed(date_entry.get("games", [])):
                if games_counted >= n_recent:
                    break
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                if g["gamePk"] == game_pk:
                    continue

                # Check pitcher handedness for this game (for platoon awareness)
                # (optional refinement — use when historical platoon data matters)
                lineups = g.get("lineups", {})
                home_id = g["teams"]["home"]["team"]["id"]
                away_id = g["teams"]["away"]["team"]["id"]

                side_key  = "homePlayers" if home_id == team_id else "awayPlayers"
                batters = lineups.get(side_key, [])

                if not batters:
                    continue

                games_counted += 1
                for b in batters:
                    pid   = b.get("id") or b.get("person", {}).get("id")
                    name  = b.get("playerName") or b.get("person", {}).get("fullName", "Unknown")
                    pos   = b.get("primaryPosition", {}).get("abbreviation", "?")
                    order = b.get("battingOrder")
                    if pid is None or order is None:
                        continue

                    slot_votes.setdefault(pid, {})
                    slot_votes[pid][order] = slot_votes[pid].get(order, 0) + 1
                    slot_names[pid] = name
                    slot_pos[pid]   = pos

        if not slot_votes:
            logger.warning(f"No historical lineup data found for team_id={team_id}")
            return []

        # For each slot 1–9, pick the player with the most appearances
        slot_to_player: Dict[int, BatterSlot] = {}
        for pid, votes in slot_votes.items():
            best_slot = max(votes, key=votes.get)
            count     = votes[best_slot]
            if best_slot not in slot_to_player or count > slot_to_player[best_slot].k_pct:
                # Re-use k_pct field temporarily to store the vote count
                slot_to_player[best_slot] = BatterSlot(
                    batting_order=best_slot,
                    player_id=pid,
                    full_name=slot_names[pid],
                    position=slot_pos[pid],
                    k_pct=count,            # vote count — overwritten later by FeatureBuilder
                )

        projected = sorted(slot_to_player.values(), key=lambda b: b.batting_order)
        logger.info(f"Projected {len(projected)} batters for team_id={team_id}")
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
            return None
        return {
            "id":       pp.get("id"),
            "fullName": pp.get("fullName"),
            "throws":   pp.get("pitchHand", {}).get("code", "R"),
        }

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
