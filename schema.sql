-- Jogos / partidas
CREATE TABLE IF NOT EXISTS matches (
    event_id        BIGINT PRIMARY KEY,
    start_ts        BIGINT,
    start_date      DATE,
    status          TEXT,
    home_id         INT,
    away_id         INT,
    home_name       TEXT,
    away_name       TEXT,
    home_score      INT,
    away_score      INT,
    tournament_id   INT,
    tournament_name TEXT,
    category_name   TEXT,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Estatísticas de jogadores por partida
CREATE TABLE IF NOT EXISTS player_stats (
    id              BIGSERIAL PRIMARY KEY,
    event_id        BIGINT REFERENCES matches(event_id) ON DELETE CASCADE,
    team_id         INT,
    player_id       INT,
    player_name     TEXT,
    position        TEXT,
    -- gerais
    minutes_played  INT,
    rating          NUMERIC(4,2),
    -- ataque
    goals           INT,
    assists         INT,
    shots_on_goal   INT,
    shots_total     INT,
    -- passe
    passes_total    INT,
    passes_accurate INT,
    key_passes      INT,
    -- defesa
    tackles         INT,
    interceptions   INT,
    clearances      INT,
    -- goleiro
    saves           INT,
    -- disciplina
    yellow_cards    INT,
    red_cards       INT,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(event_id, player_id)
);

-- Liga/torneio para filtrar ligas principais
CREATE TABLE IF NOT EXISTS tournaments (
    tournament_id   INT PRIMARY KEY,
    name            TEXT,
    category_name   TEXT,
    tracked         BOOLEAN DEFAULT TRUE
);

-- Controle de coleta (evita reprocessar o mesmo dia)
CREATE TABLE IF NOT EXISTS collection_log (
    log_date        DATE PRIMARY KEY,
    collected_at    TIMESTAMPTZ DEFAULT NOW(),
    events_found    INT,
    stats_collected INT
);
