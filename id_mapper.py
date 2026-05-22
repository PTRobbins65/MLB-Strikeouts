"""
MLB Strikeout Pipeline — Player ID Mapper
Downloads the Chadwick Bureau register once to build a MLBAM ↔ FanGraphs
IDfg cross-reference table, saved to data/player_id_map.parquet.

Usage
-----
    python id_mapper.py            # build / refresh the map

    from id_mapper import IdMapper
    mapper = IdMapper()
    fg_id  = mapper.mlbam_to_fg(477132)   # -> FanGraphs IDfg
    mlbam  = mapper.fg_to_mlbam(3973)     # -> MLBAM ID
"""

import logging
from typing import Optional

import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)

MAP_PATH = DATA_DIR / "player_id_map.parquet"


def _pybaseball():
    try:
        import pybaseball as pb
        return pb
    except ImportError:
        raise ImportError("pybaseball is required: pip install pybaseball")


class IdMapper:
    """
    Thin wrapper around the Chadwick Bureau player register.
    Maps MLBAM player IDs (used by Statcast / StatsAPI) to FanGraphs IDfg.
    The register is downloaded once and cached to disk.
    """

    def __init__(self, force_refresh: bool = False):
        self._map = self._load_or_build(force_refresh)

    def mlbam_to_fg(self, mlbam_id: int) -> Optional[int]:
        """Return FanGraphs IDfg for a given MLBAM player ID, or None."""
        row = self._map[self._map["key_mlbam"] == mlbam_id]
        if row.empty:
            return None
        val = row["key_fangraphs"].values[0]
        return None if pd.isna(val) else int(val)

    def fg_to_mlbam(self, fg_id: int) -> Optional[int]:
        """Return MLBAM ID for a given FanGraphs IDfg, or None."""
        row = self._map[self._map["key_fangraphs"] == fg_id]
        if row.empty:
            return None
        val = row["key_mlbam"].values[0]
        return None if pd.isna(val) else int(val)

    def get_map(self) -> pd.DataFrame:
        return self._map.copy()

    @staticmethod
    def _load_or_build(force_refresh: bool) -> pd.DataFrame:
        if MAP_PATH.exists() and not force_refresh:
            logger.info(f"Loading cached ID map: {MAP_PATH.name}")
            return pd.read_parquet(MAP_PATH)

        pb = _pybaseball()
        logger.info("Downloading Chadwick Bureau player register (~2 MB, one-time)...")
        register = pb.chadwick_register()

        needed = ["key_mlbam", "key_fangraphs", "name_first", "name_last"]
        cols = [c for c in needed if c in register.columns]
        df = register[cols].dropna(subset=["key_mlbam", "key_fangraphs"]).copy()
        df["key_mlbam"]     = df["key_mlbam"].astype(int)
        df["key_fangraphs"] = df["key_fangraphs"].astype(int)
        df = df.drop_duplicates(subset=["key_mlbam"])

        MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(MAP_PATH, index=False)
        logger.info(f"ID map saved: {len(df):,} players -> {MAP_PATH.name}")
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    mapper = IdMapper(force_refresh=True)
    df = mapper.get_map()
    print(f"ID map: {len(df):,} players with both MLBAM and FanGraphs IDs")
    # Quick smoke test with a known player
    KERSHAW_MLBAM = 477132
    fg = mapper.mlbam_to_fg(KERSHAW_MLBAM)
    print(f"Kershaw (MLBAM {KERSHAW_MLBAM}) -> FanGraphs IDfg {fg}")
