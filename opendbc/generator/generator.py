#!/usr/bin/env python2
import os
import re

cur_path = os.path.dirname(os.path.realpath(__file__))
generator_path = os.path.join(cur_path, '../')

for dir_name, _, _ in os.walk(cur_path):
    if dir_name == cur_path:
        continue

    print dir_name

    for filename in os.listdir(dir_name):
        if filename.startswith('_'):
            continue

        print filename
        dbc_file = open(os.path.join(dir_name, filename)).read()
        dbc_file = '\nCM_ "%s starts here"\n' % filename + dbc_file

        includes = re.finditer(r'CM_ "IMPORT (.*?)"', dbc_file)
        for include in includes:
            dbc_file = dbc_file.replace(include.group(0), '')
            include_path = os.path.join(dir_name, include.group(1))

            # Load included file
            include_file = open(include_path).read()
            include_file = '\n\nCM_ "Imported file %s starts here"\n' % include.group(1) + include_file
            dbc_file = include_file + dbc_file

        dbc_file = 'CM_ "AUTOGENERATED FILE, DO NOT EDIT"\n' + dbc_file

        output_filename = filename.replace('.dbc', '_generated.dbc')
        output_dbc_file = open(os.path.join(generator_path, output_filename), 'w')
        output_dbc_file.write(dbc_file)
        output_dbc_file.close()
