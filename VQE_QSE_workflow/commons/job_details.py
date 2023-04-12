def write_job_details(logfile,isgit=False):
    import qiskit
    import subprocess
    from   datetime import datetime
    import numpy as np
    import scipy as sp
    import pyscf
    import platform

    dateTimeObj = datetime.now()
    logfile.write("date and time "+str(dateTimeObj)+"\n")
    logfile.write("\n")
    logfile.write("PySCF  version %s \n" % pyscf.__version__)
    logfile.write("Python version %s \n" % platform.python_version())
    logfile.write("numpy  version %s \n" % np.__version__)
    logfile.write("scipy  version %s \n" % sp.__version__)
    logfile.write("\n")
    qiskit_dict = qiskit.__qiskit_version__
    logfile.write("qiskit version \n")
    logfile.write(str(qiskit_dict))
    logfile.write("\n")
    if(isgit):
       label = subprocess.check_output(['git','log','-1','--format="%H"']).strip()
       logfile.write("git commit "+str(label)+"\n")
       logfile.write("\n")

