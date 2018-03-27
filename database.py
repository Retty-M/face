# coding=utf-8
import sqlite3


class DB:

    def __init__(self, dbfile):
        self.conn = sqlite3.connect(dbfile)
        self.cursor = self.conn.cursor()
        if self.is_exist_table() is False:
            self.create_table_user()

    def is_exist_table(self):
        print('Check table user')
        check_table_sql = '''SELECT count(*) FROM sqlite_master WHERE type='table' AND name='user';'''
        self.cursor.execute(check_table_sql)
        values = self.cursor.fetchone()
        if values[0] > 0:
            return True
        else:
            return False

    def create_table_user(self):
        print('Create table user')
        create_table_sql = '''CREATE TABLE `user` (
                                `name`	TEXT NOT NULL UNIQUE,
                                `age`	INTEGER,
                                `duty`	TEXT,
                                `department`	TEXT,
                                `secret`	TEXT,
                                PRIMARY KEY(name)
                                );'''
        self.cursor.execute(create_table_sql)
        self.conn.commit()

    def add_data(self, data):
        if data is not None:
            print('Add data in user')
            add_data_sql = '''INSERT INTO user values ('%s', %d, '%s', '%s', '%s');''' % \
                           (data['name'], data['age'], data['duty'], data['department'], data['secret'])
            self.cursor.execute(add_data_sql)
            self.conn.commit()

    def fetchone(self, name):
        if name is not None:
            fetch_one_sql = '''SELECT * FROM user WHERE name='%s';''' % name
            self.cursor.execute(fetch_one_sql)
            values = self.cursor.fetchone()
            # print(values)
            return values

    def close(self):
        self.cursor.close()
        self.conn.close()


# db = DB('./data/info')
# db.add_data({'name':'sd', 'age':12, 'duty':'', 'department':'', 'secret':''})
# db.fetchone("sd")
# db.close()

