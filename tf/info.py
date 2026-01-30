#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 13:38:22 2023

@author: hessel
"""

DEBUGLEVEL = 3

from datetime import datetime

class Info:
    def __init__(self, logfile="log.txt", exit_on_error = True):
        self.logfile = logfile
        self.exit_on_error = exit_on_error
        self.printers={}
        self.printers["info"] = print
        self.printers["warning"] = print
        self.printers["error"] = print
        self.printers["debug"] = print

    def set_printer(self, info_type, print_function):
        self.printer[info_type.lower] = print_function
        
    def set_log(self, path):
        self.logfile = path
        
    def log(self, message):
        file = open(self.logfile, "a")
        file.write(datetime.utcnow().isoformat(timespec="milliseconds")+"Z ")
        file.write(message)
        file.write("\n")
        file.close()
        
    def info(self, *args):
        message = " ".join(("Info:",) + args)
        if "info" in self.printers:
            printer = self.printers["info"]
            if printer:        
                printer(message)
        if self.logfile is not None:
            self.log(message)

    def warning(self, *args):
        message = " ".join(("Warning:",) + args)
        if "warning" in self.printers:
            printer = self.printers["warning"]
            if printer:        
                printer(message)
        if self.logfile is not None:
            self.log(message)

    def error(self, *args):
        message = " ".join(("Error:",) + args)
        if "error" in self.printers:
            printer = self.printers["error"]
            if printer:        
                printer(message)
        if self.logfile is not None:
            self.log(message)
        if self.exit_on_error:
            import sys
            sys.exit(1)
    
    def debug(self, level, *args):
        if level < DEBUGLEVEL:
            return
        message = " ".join((f"Debug ({level}):",) + args)
        if "debug" in self.printers:
            printer = self.printers["warning"]
            if printer:
                printer(message)


_info = Info()

info = _info.info
warning = _info.warning
error = _info.error
debug = _info.debug
set_printer = _info.set_printer
set_log = _info.set_log


if __name__ == "__main__":
    info("1", "%d" % 2, f"{DEBUGLEVEL}")
    info("Debuglevel", "%s" % ("is"), f"{DEBUGLEVEL}.")
    debug(1, "not shown!")
    debug(3, "Spam!")
    warning("error %s..." % "approaching")
    error("Goodbye.")
    