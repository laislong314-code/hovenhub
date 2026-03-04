#!/usr/bin/env python3
"""Migration idempotente: adiciona colunas sofa_xg_home/sofa_xg_away em match_analyses.

Uso:
  python scripts/migrate_add_sofa_xg.py --db data/sports_ev.db

Seguro para produção:
- Não remove nada.
- Só adiciona colunas se ainda não existirem.
"""

import argparse
import sqlite3
from pathlib import Path

def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]  # row[1] = name
    return column in cols

def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl_type: str):
    if column_exists(conn, table, column):
        print(f"[OK] {table}.{column} já existe")
        return
    print(f"[ADD] {table}.{column} ({ddl_type})")
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Caminho do sqlite .db (ex: data/sports_ev.db)")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB não encontrado: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        add_column_if_missing(conn, "match_analyses", "sofa_xg_home", "REAL")
        add_column_if_missing(conn, "match_analyses", "sofa_xg_away", "REAL")
        conn.commit()
        print("[DONE] Migration concluída")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
