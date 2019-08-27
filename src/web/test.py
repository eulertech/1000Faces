from flask import Flask,request
from flask_restful import Resource, Api
from json import dumps
from flask import jsonify
from getKey import getKey
from teambuilderAPI import buildTeam
app = Flask(__name__)
api = Api(app)


class Employees_Name(Resource):
    def get(self, employee):
        result = {'employeeName': {getKey():employee}}
        return jsonify(result)

class teambuilder(Resource:
    def get(self, text):
        return

api.add_resource(Employees_Name, '/employees/<employee>')

if __name__ == '__main__':
    app.run(port='5002')
