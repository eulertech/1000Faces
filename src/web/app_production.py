from flask import Flask,request
from flask_restful import Resource, Api
from json import dumps
from flask import jsonify
from getKey import getKey
from teambuilderAPI import buildTeam
from util import get_employee # get employee with employee list and return json
from util import get_project # get project details with project List
from util import get_node # produce note, link with list of employee id
app = Flask(__name__)
api = Api(app)

@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/teambuilder', methods=['POST'])
def my_form_post():
    text = request.form['q']
    eIdList = buildTeam(text)
    result = get_node(eIdList)
    #result = {'teamMembers': eIdList}
    return jsonify(result)

class Employees_Name(Resource):
    # this only take a single Id
    def get(self, eId):
        result = get_employee(list(str(eId)))
        return jsonify(result)

class projectrecommendation(Resource):
    def get(self, eId):
        projectList = findTopKSimilarProject(eId)
        result = get_project(projectList)
        #result = {'projectsRecommended': projectList}
        return jsonify(result)

class userrecommendation(Resource):
    def get(self, eId):
        userList = findTopKSimilarEmployee(eId)
        result = get_employee(userList)
        return jsonify(result)

#api.add_resource(teambuilder, '/teambuilder/<inputString>')
api.add_resource(projectrecommendation, '/projectrecommendation/<eId>')
api.add_resource(userrecommendation,'/userrecommendation/<eId>')
api.add_resource(Employees_Name, '/employees/<eId>')

if __name__ == '__main__':
    app.run(port='5002')
