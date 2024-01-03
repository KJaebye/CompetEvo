import competevo
import gym_compete

import gymnasium as gym
from config.config import Config
import argparse

from competevo.evo_envs.agents.dev_ant import DevAnt
from lxml import etree
import numpy as np

ant = DevAnt(1, xml_path="competevo/evo_envs/assets/evo_ant_body_base2.xml")

a = np.ones(20)*3


def multiply_str(s, m):
    res = [str(float(x) * m) for x in s.split()]
    res_str = ' '.join(res)
    return res_str

agent_body = ant.tree.find('body')
for body in agent_body.iter('body'):
    cur_name = body.get('name')

    # 1
    if cur_name == "1":
        geom = body.find('geom') #1
        if geom is not None:
            p = geom.get("fromto")
            p = multiply_str(p, a[0])
            geom.set("fromto", p)

    if cur_name == "11":
        p = body.get("pos")
        p = multiply_str(p, a[0])
        body.set("pos", p)

        geom = body.find('geom') #11
        p = geom.get("size")
        p = multiply_str(p, a[1])
        geom.set("size", p)

        if geom is not None:
            p = geom.get("fromto")
            p = multiply_str(p, a[2])
            geom.set("fromto", p)

    if cur_name == "111":
        p = body.get("pos")
        p = multiply_str(p, a[2])
        body.set("pos", p)

        geom = body.find('geom') #111
        p = geom.get("size")
        p = multiply_str(p, a[3])
        geom.set("size", p)

        if geom is not None:
            p = geom.get("fromto")
            p = multiply_str(p, a[4])
            geom.set("fromto", p)
    
    # 2
    if cur_name == "2":
        geom = body.find('geom') #2
        if geom is not None:
            p = geom.get("fromto")
            p = multiply_str(p, a[5])
            geom.set("fromto", p)

    if cur_name == "12":
        p = body.get("pos")
        p = multiply_str(p, a[5])
        body.set("pos", p)

        geom = body.find('geom') #12
        p = geom.get("size")
        p = multiply_str(p, a[6])
        geom.set("size", p)

        if geom is not None:
            p = geom.get("fromto")
            p = multiply_str(p, a[7])
            geom.set("fromto", p)

    if cur_name == "112":
        p = body.get("pos")
        p = multiply_str(p, a[7])
        body.set("pos", p)

        geom = body.find('geom') #112
        p = geom.get("size")
        p = multiply_str(p, a[8])
        geom.set("size", p)

        if geom is not None:
            p = geom.get("fromto")
            p = multiply_str(p, a[9])
            geom.set("fromto", p)

    # 3
    if cur_name == "3":
        geom = body.find('geom') #3
        if geom is not None:
            p = geom.get("fromto")
            p = multiply_str(p, a[10])
            geom.set("fromto", p)

    if cur_name == "13":
        p = body.get("pos")
        p = multiply_str(p, a[10])
        body.set("pos", p)

        geom = body.find('geom') #13
        p = geom.get("size")
        p = multiply_str(p, a[11])
        geom.set("size", p)

        if geom is not None:
            p = geom.get("fromto")
            p = multiply_str(p, a[12])
            geom.set("fromto", p)

    if cur_name == "113":
        p = body.get("pos")
        p = multiply_str(p, a[12])
        body.set("pos", p)

        geom = body.find('geom') #113
        p = geom.get("size")
        p = multiply_str(p, a[13])
        geom.set("size", p)

        if geom is not None:
            p = geom.get("fromto")
            p = multiply_str(p, a[14])
            geom.set("fromto", p)

    # 4
    if cur_name == "4":
        geom = body.find('geom') #4
        if geom is not None:
            p = geom.get("fromto")
            p = multiply_str(p, a[15])
            geom.set("fromto", p)

    if cur_name == "14":
        p = body.get("pos")
        p = multiply_str(p, a[15])
        body.set("pos", p)

        geom = body.find('geom') #14
        p = geom.get("size")
        p = multiply_str(p, a[16])
        geom.set("size", p)

        if geom is not None:
            p = geom.get("fromto")
            p = multiply_str(p, a[17])
            geom.set("fromto", p)

    if cur_name == "114":
        p = body.get("pos")
        p = multiply_str(p, a[17])
        body.set("pos", p)

        geom = body.find('geom') #114
        p = geom.get("size")
        p = multiply_str(p, a[18])
        geom.set("size", p)

        if geom is not None:
            p = geom.get("fromto")
            p = multiply_str(p, a[19])
            geom.set("fromto", p)

agent_actuator = ant.tree.find('actuator')
for motor in agent_actuator.iter("motor"):
    cur_name = motor.get("name").split('_')[0]

    if cur_name == "11":
        p = motor.get("gear")
        p = multiply_str(p, a[1])
        motor.set("gear", p)
    
    if cur_name == "111":
        p = motor.get("gear")
        p = multiply_str(p, a[3])
        motor.set("gear", p)
    
    if cur_name == "12":
        p = motor.get("gear")
        p = multiply_str(p, a[6])
        motor.set("gear", p)
    
    if cur_name == "112":
        p = motor.get("gear")
        p = multiply_str(p, a[8])
        motor.set("gear", p)
    
    if cur_name == "13":
        p = motor.get("gear")
        p = multiply_str(p, a[11])
        motor.set("gear", p)
    
    if cur_name == "113":
        p = motor.get("gear")
        p = multiply_str(p, a[13])
        motor.set("gear", p)

    if cur_name == "14":
        p = motor.get("gear")
        p = multiply_str(p, a[16])
        motor.set("gear", p)
    
    if cur_name == "114":
        p = motor.get("gear")
        p = multiply_str(p, a[18])
        motor.set("gear", p)

print(etree.tostring(ant.tree, pretty_print=True).decode('utf-8'))
