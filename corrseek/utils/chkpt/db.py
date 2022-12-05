from contextlib import contextmanager
import datetime as dt
import json
import os
import sqlite3 as s


class CHKPT:
    def __init__(self, db_dir, fname):
        if not os.path.exists(db_dir):
            raise ValueError(
                f"path provided does not exists {db_dir}"
            )
        self.db_abs_dir = os.path.join(db_dir, fname)
        if not os.path.isfile(self.db_abs_dir):
            self.initialize()
        
    @contextmanager
    def _conn(self):
        try:
            conn = s.connect(self.db_abs_dir)
            yield conn
        finally:
            conn.close()
    
    def initialize(self):
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                '''CREATE TABLE IF NOT EXISTS model (
                        id TEXT PRIMARY KEY,
                        modelname TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        inuse INT NOT NULL,
                        parameters TEXT NOT NULL
                    )
                '''
            )
    
    def save_model(self, payload):
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                '''UPDATE model
                    SET inuse = 0
                    WHERE
                        modelname = ?
                ''',
                (
                    payload['modelname'],
                )
            )
            cur.execute(
                f'''INSERT INTO model (
                    modelname,
                    timestamp,
                    inuse,
                    parameters
                ) VALUES (
                    '{payload['modelname']}',
                    '{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}',
                    1,
                    '{json.dumps(payload['parameters'])}'
                )
                '''
            )
            conn.commit()

    def retrieve_model(self, modelname):
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                '''SELECT parameters
                    FROM model
                    WHERE
                        modelname = ?
                        AND inuse = 1
                ''',
                (
                    modelname,
                )
            )
            ret = cur.fetchone()[0]
            return json.loads(ret)