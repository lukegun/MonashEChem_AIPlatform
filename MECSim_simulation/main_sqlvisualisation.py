
import psycopg2
import scipy as sci
from scipy.special import jv, yn    # jv = first order, yn =second kind
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import NN_posttraining_mod as NNt_post
import time

print("FUCK")
dname = "timeseriesFTACV"#"DC_db_kf"

#help("modules")
try:
    connection = psycopg2.connect(user = "postgres",
                                  password = "password",
                                  host = "host",
                                  port = "port",
                                  database = dname)

    cursor = connection.cursor()
    b = np.random.randint(0,6)
    ki = 24
    i = b*1*ki
    i = ki*0
    #i =240*(b)
    N = np.random.randint(0+i,ki+i)#(10016,20024)
    #N = 17152 # EC low diff
    #N = 4000
    print("Reaction ID: " + str(N))
    #N = 1#18446

    #print(connection.get_dsn_parameters(),"\n")
    #"HarmCol6"
    curr = cursor.execute("""SELECT "HarmCol0" FROM "HarmTab" WHERE "Reaction_ID" = %s""", (N,))

    for table in cursor.fetchall():
        #print(table)
        x = table[0]

    para = cursor.execute("""SELECT * FROM "Simulatedparameters" WHERE "Reaction_ID" =  %s""", (N,))
    for table in cursor.fetchall():
        para = table
    print(para)
    #sqlcom = """INSERT INTO time(test,"Ctime") VALUES( %()s, CURRENT_TIMESTAMP )"""

except (Exception, psycopg2.Error) as error:
    print("error,", error)
finally:
    # This is needed for the
    if(connection):
        cursor.close()
        connection.close()

    print("it did something")
cursor.close()
connection.close()

plt.plot(x)
#plt.plot(tot,x[:int(len(x)/4)])
plt.savefig("fucker.png")
plt.close()

