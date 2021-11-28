# -*- coding: utf-8 -*-
from __future__ import print_function
import logging
import socket

DEBUG =0

SPARSE=False
WARMUP=True
DELAY_COMM=1

if SPARSE:
    PREFIX='compression'
else:
    PREFIX='baseline'
if WARMUP:
    PREFIX=PREFIX+'-gwarmup'

TENSORBOARD=True
PROFILING_NORM=False
PROFILING_GRAD=False
PROFILING=False

hostname = socket.gethostname() 
logger = logging.getLogger(hostname)

if DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)

