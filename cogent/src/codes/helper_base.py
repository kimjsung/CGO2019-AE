# to write a line--- code
def codes_a_line(f, num_tabs, str_code, opt_linebreak):
    #
    for each_tab in range(num_tabs):
        f.write("\t")
    #
    f.write(str_code)
    #
    if opt_linebreak == 1:
        f.write("\n")
