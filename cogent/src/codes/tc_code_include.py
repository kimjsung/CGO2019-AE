#
#
#
def tc_code_include(f):
    # include
    f.write("// created by tc_code_include() in tc_code_include.py\n")
    f.write("#include <stdio.h>\n")
    f.write("#include <stdlib.h>\n")
    f.write("#include <unistd.h>\n")
    f.write("#include <sys/time.h>\n")
    f.write("#include <locale.h>\n")
    f.write("#include <algorithm>\n")       # g++
    f.write("using namespace std;\n")
    f.write("\n")