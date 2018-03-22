import apsw
from multiprocessing import Lock

db = apsw.Connection("DB.sqlite")
dbcur = db.cursor()
l = Lock()