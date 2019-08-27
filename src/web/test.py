from flask import Flask,request
from flask_restful import Resource, Api
from json import dumps
from flask import jsonify
app = Flask(__name__)
api = Api(app)

class Employees_Name(Resource):
    def get(self, employee):
        result = {'employeeName': {'name':employee}}
        return jsonify(result)

api.add_resource(Employees_Name, '/employees/<employee>')

if __name__ == '__main__':
    app.run(port='5002')


