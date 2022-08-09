from urllib import request
import json
import pandas as pd
import sqlite3
from pathlib import Path

## SQLITE3 database functions
# Connect with database
def create_connection(db_file_name):
    conn = None

    try:
        conn = sqlite3.connect(db_file_name)
    except sqlite3.Error as e:
        print(e)

    return conn

# Create new table in database
def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)


# Create database table for patients
def create_patients_table(conn):
    sql_create_patients_table = """CREATE TABLE IF NOT EXISTS patients (
        patient_id text,
        sexe text,
        patient_train_or_test text,
        diagnosis text NOT NULL,
        nr_of_sessions integer NOT NULL,
        PRIMARY KEY (patient_id)
    );"""

    create_table(conn, sql_create_patients_table)

# Create database table for sessions
def create_sessions_table(conn):
    sql_create_sessions_table = """CREATE TABLE IF NOT EXISTS sessions (
        patient_id text,
        session_id text,
        electrode_setup text,
        date text,
        nr_of_tokens integer NOT NULL,
        PRIMARY KEY (patient_id, session_id),
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
    );"""

    create_table(conn, sql_create_sessions_table)

# Create database table for tokens
def create_tokens_table(conn):
    sql_create_tokens_table = """CREATE TABLE IF NOT EXISTS tokens (
        patient_id text,
        session_id text,
        token_id text,
        url text NOT NULL,
        file_name text NOT NULL,
        file_path text NOT NULL,
        sampling_freq integer NOT NULL,
        nr_of_samples integer NOT NULL,
        len_of_samples integer NOT NULL,
        recording_duration float,
        nr_of_chs integer NOT NULL,
        eeg_chs text NOT NULL,
        non_eeg_chs text,
        PRIMARY KEY (patient_id, session_id, token_id),
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id),
        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
    );"""

    create_table(conn, sql_create_tokens_table)

# Add patient record to database
def create_patient(conn, patient):
    sql = """INSERT INTO patients(patient_id, sexe, patient_train_or_test, diagnosis, nr_of_sessions)
             VALUES(?,?,?,?,?)"""

    cur = conn.cursor()
    cur.execute(sql, patient)
    return cur.lastrowid
    
# Add sessions record to database
def create_session(conn, session):
    sql = """INSERT INTO sessions(session_id, patient_id, electrode_setup, date, nr_of_tokens)
             VALUES(?,?,?,?,?)"""

    cur = conn.cursor()
    cur.execute(sql, session)
    return cur.lastrowid

# Add token record to database
def create_token(conn, token):
    sql = """INSERT INTO tokens(token_id, session_id, patient_id, url, file_name, file_path, sampling_freq, nr_of_samples, len_of_samples, recording_duration, nr_of_chs, eeg_chs, non_eeg_chs)
             VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)"""

    cur = conn.cursor()
    cur.execute(sql, token)
    return cur.lastrowid

# Select patient records by id
def select_patients_by_id(conn, patient_id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,))

    rows = cur.fetchall()

    if len(rows) == 0:
        return False
    else:
        #for row in rows:
        #    print(row)
        return True
        
# Select session records by id
def select_sessions_by_id(conn, patient_id, session_id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE patient_id=? AND session_id=? ", (patient_id,session_id,))

    rows = cur.fetchall()

    if len(rows) == 0:
        return False
    else:
        #for row in rows:
        #    print(row)
        return True
        
# Select token records by id
def select_tokens_by_id(conn, patient_id, session_id, token_id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM tokens WHERE patient_id=? AND session_id=? AND token_id=?", (patient_id,session_id,token_id,))

    rows = cur.fetchall()

    if len(rows) == 0:
        return False
    else:
        #for row in rows:
        #    print(row)
        return True

# Auxillary function to update patient's nr_of_sessions to correct (total) number
def update_patient_nr_of_sessions(conn, patient_id, nr_of_sessions):
    cur = conn.cursor()
    cur.execute("SELECT nr_of_sessions FROM patients WHERE patient_id=?", (patient_id,))
    
    nr_of_sessions_old = cur.fetchall()[0][0]

    sql = """UPDATE patients SET nr_of_sessions = ? WHERE patient_id = ?"""
    cur.execute(sql, (nr_of_sessions_old + nr_of_sessions, patient_id))

## MAIN FUNCTION
# Connect with database
conn = create_connection("/home/jupyter/time_series_transfer_learning/files/TUAB_files/eeg_recordings_TUAB.db")

# Create tables (if they do not exist yet)
if conn is not None:
    create_patients_table(conn)
    create_sessions_table(conn)
    create_tokens_table(conn)

# Read in Temple University Hospital file structure as defined in JSON file
with open('data/JSON_files/TUAB/TUAB_type.json', 'r') as f:
    data_structure = json.load(f)

# Iterate over data structure to fetch download link and metadata info;
# download file and write metadata to database
for train_or_test in data_structure.keys():
    for diagnosis in data_structure[train_or_test].keys():
        for electrode_setup in data_structure[train_or_test][diagnosis].keys():
            for patient_id in data_structure[train_or_test][diagnosis][electrode_setup].keys():
                for session_id in data_structure[train_or_test][diagnosis][electrode_setup][patient_id].keys():
                    for token_id in data_structure[train_or_test][diagnosis][electrode_setup][patient_id][session_id].keys():

                        edf_url = data_structure[train_or_test][diagnosis][electrode_setup][patient_id][session_id][token_id]
                        #print(edf_url)
                        # Check if (patient_id, session_id, token_id) is not already in database
                        if select_tokens_by_id(conn, patient_id, session_id[:4], token_id):
                            continue

                        # Define password manager and authentication handler to log in to TUH EEG Corpus download page
                        password_mgr = request.HTTPPasswordMgrWithDefaultRealm()
                        password_mgr.add_password(None, edf_url, "nedc", "nedc_resources")
                        auth_handler = request.HTTPBasicAuthHandler(password_mgr)

                        # Open directory url with defined authentication handler
                        url_opener = request.build_opener(auth_handler)
                        req = request.Request(edf_url)

                        # Retrieve edf file's first 10000 bytes 
                        edf_bytes = url_opener.open(req).read()
                        
                        # Read in patient information
                        patient_header = edf_bytes[8:88].decode("utf-8")
                        patient_nr = str((patient_header[0:8]))
                        patient_sexe = str((patient_header[9:10]))
                        #patient_age = int(patient_header[36:38])
                        
                        # Read in information from url
                        edf_url_parts = edf_url.split('/')
                        edf_file_name = patient_nr + '_' + edf_url_parts[14] + '.edf'
                        edf_file_path = "/" + "/".join(edf_url_parts[9:14]) + "/"
                        patient_diagnosis = edf_url_parts[9]
                        recording_setup = edf_url_parts[10]
                        recording_id = edf_url_parts[14][:-4]
                        print(f"recording_id({recording_id})")

                        # Add code to download file and save to specified path
                        Path("/home/jupyter/time_series_transfer_learning/files/TUAB_files" + edf_file_path).mkdir(parents=True, exist_ok=True)
                        with open("/home/jupyter/time_series_transfer_learning/files/TUAB_files" + edf_file_path + edf_file_name, 'wb') as f:
                            f.write(edf_bytes)


                        # Read in session information
                        session_information = edf_url_parts[13]
                        session_nr = session_information[:4]

                        # Read in session date information
                        session_data_information = (edf_bytes[168:184]).decode("utf-8")
                        session_date_year = session_data_information[0:2]
                        session_date_month = session_data_information[3:5]
                        session_date_day = session_data_information[6:8]
                        session_date_hour = session_data_information[8:10]
                        session_date_minute = session_data_information[11:13]
                        session_date_second = session_data_information[14:16]

                        session_date = f"{session_date_year}/{session_date_month}/{session_date_day} {session_date_hour}:{session_date_minute}:{session_date_second}"

                        # Read in token information
                        token_information = edf_url_parts[14]
                        token_nr = token_information[-8:-4]

                        # Read in records information
                        records_header = edf_bytes[236:256].decode("utf-8")
                        nr_of_samples = int(records_header[0:8])
                        len_of_samples = float(records_header[8:16])
                        nr_of_chs = int(records_header[16:20])

                        # Read in sample freq 
                        sampling_freq = edf_bytes[256+(216*nr_of_chs):256+(216*(nr_of_chs))+8].decode("utf-8")
                        sampling_freq = int(sampling_freq.strip())

                        # Extract addiitonal recording info from records information
                        recording_duration = int(nr_of_samples * len_of_samples)
                        nr_of_samples = int(recording_duration * sampling_freq)

                        # Read in channel information
                        chs = edf_bytes[256:256+(nr_of_chs*16)].decode("utf-8")
                        chs = [chs[16*ch_idx:16*(ch_idx+1)].strip() for ch_idx in range(nr_of_chs)] 
                        nr_of_chs = len(chs)

                        # Filter channel names channel names
                        eeg_chs = [ch for ch in chs if ch[:3] == "EEG"]
                        eeg_chs.sort()

                        non_eeg_chs = [ch for ch in chs if ch[:3] != "EEG"]
                        non_eeg_chs.sort()


                        # Add new token record to database
                        #print("token_nr " + token_nr + " session_nr " + session_nr + " patient_nr " + patient_nr)
                        print(edf_file_name)
                        token = (token_id, session_id[:4], patient_id, edf_url, edf_file_name, edf_file_path, sampling_freq, nr_of_samples, len_of_samples, recording_duration, nr_of_chs, str(eeg_chs), str(non_eeg_chs))
                        create_token(conn, token)
                        conn.commit()



                    # Add new session record to database
                    if select_sessions_by_id(conn, patient_id, session_id[:4]):
                        continue

                    nr_of_tokens = len(list(data_structure[train_or_test][diagnosis][electrode_setup][patient_id][session_id].keys()))
                    session = (session_id[:4], patient_id, electrode_setup, session_date, nr_of_tokens)
                    create_session(conn, session)
                    conn.commit()

                # Add new patient record to database or update existing one
                nr_of_sessions = len(list(data_structure[train_or_test][diagnosis][electrode_setup][patient_id].keys()))

                if select_patients_by_id(conn, patient_id):
                    update_patient_nr_of_sessions(conn, patient_id, nr_of_sessions)
                else:   
                    patient = (patient_id, patient_sexe, train_or_test, diagnosis, nr_of_sessions)
                    create_patient(conn, patient)
                conn.commit()
            
# Commit any uncommitted changes and close database connection
if conn:
    conn.commit()
    conn.close()


