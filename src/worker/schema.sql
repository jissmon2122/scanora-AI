-- D1 Database schema for Scanora
CREATE TABLE IF NOT EXISTS scans (
  id TEXT PRIMARY KEY,
  user_email TEXT,
  file_name TEXT NOT NULL,
  file_size INTEGER NOT NULL,
  uploaded_at TEXT NOT NULL,
  patient_id TEXT,
  patient_age INTEGER,
  laterality TEXT DEFAULT 'right',
  mammography_view TEXT DEFAULT 'MLO',
  notes TEXT,
  birads_category TEXT,
  birads_description TEXT,
  malignancy_score REAL,
  benign_score REAL,
  density TEXT,
  density_description TEXT,
  findings_json TEXT,
  recommendation TEXT,
  analyzed_at TEXT,
  image_url TEXT
);

CREATE INDEX IF NOT EXISTS idx_scans_user_email ON scans(user_email);
CREATE INDEX IF NOT EXISTS idx_scans_uploaded_at ON scans(uploaded_at DESC);
