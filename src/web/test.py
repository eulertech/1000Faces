from __future__ import print_function
from flask import Flask,request
from flask_restful import Resource, Api
from flask import Flask, render_template, make_response
from flask import redirect, request, jsonify, url_for
from json import dumps
from flask import jsonify
from getKey import getKey
import sys
import os
import os.path
import psycopg2
import json
from teambuilderAPI import buildTeam
from teambuilderAPI import preprocess
from projectRecomAPI import findTopKSimilarProject
from userRecomAPI import findTopKSimilarEmployee
app = Flask(__name__)
api = Api(app)


@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/teambuilder', methods=['POST'])
def my_form_post():
    text = request.form['q']
    processed_text = text.upper()
    print(processed_text)
    eIdList = buildTeam(processed_text)
    result = {'teamMembers': eIdList}
    return jsonify(result)

class Employees_Name(Resource):
    def get(self, eId):
        result = {'employeeName': getKey(eId)}
        return jsonify(result)

class projectrecommendation(Resource):
    def get(self, eId):
        projectList = findTopKSimilarProject(eId)
        result = {'projectsRecommended': projectList}
        return jsonify(result)

class userrecommendation(Resource):
    def get(self, eId):
        userList = findTopKSimilarEmployee(eId)
        result = {'usersRecommended': userList}
        return jsonify(result)

#api.add_resource(teambuilder, '/teambuilder/<inputString>')
api.add_resource(projectrecommendation, '/projectrecommendation/<eId>')
api.add_resource(userrecommendation,'/userrecommendation/<eId>')
api.add_resource(Employees_Name, '/employees/<eId>')

if __name__ == '__main__':
    app.run(port='5002')
