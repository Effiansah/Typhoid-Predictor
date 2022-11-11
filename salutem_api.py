from operator import concat
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import json
from fyp.comp_desc.cd import generate_molecular_descriptors as get_desc
import subprocess


app = Flask(__name__)

CORS(app)

# creating api object
api = Api(app)

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('type')
parser.add_argument('compoundData')

# Define how the api will respond to the post requests
class typhoidClassifier(Resource):
    def post(self):
        args = parser.parse_args()
        
        pred_type = args['type']
        comp_smiles = args['compoundData']

        
        if comp_smiles is None:
            return jsonify({"output": "Error Parsing Compound "+args['compoundData']+" "+comp_smiles})

        
        prediction = ""

    

        desc_filename = get_desc(smiles=comp_smiles)
  
        error = False
        
        try:
            df = pd.read_csv(desc_filename, sep="\t", header=0)
           
            subprocess.run(f"rm ./{desc_filename}", shell=True)
            df.drop(columns=["Number"], axis=1, inplace=True)

            return jsonify(df);
            with open('modules/Results/RF_model.sav', 'rb') as rf:
                rf_model = pickle.load(rf)
            
            #with open('modules/Results/naive_model.sav', 'rb') as nm:
            #    nm_model = pickle.load(nm)
            
            #with open('modules/Results/NB_model.sav', 'rb') as nb:
            #    nb_model = pickle.load(nb)
            
            #with open('modules/Results/SV.sav', 'rb') as sv:
            #    sv_model = pickle.load(sv)
                
            #if(pred_type=='LR'):
                #prediction = lr_model.predict(comp_struct)
            #if(pred_type=='NN'):
            #    prediction = nn_model.predict(comp_struct)
            if(pred_type=='RF'):
                prediction = rf_model.predict(df)
            #if(pred_type=='SV'):
            #    prediction = sv_model.predict(comp_mol)
            #if(pred_type=='NB'):
            #    prediction = nb_model.predict(comp_mol)
            #if(pred_type=='NM'):
            #    prediction = nm_model.predict(comp_mol)
            
            #return jsonify(prediction.tolist())
            
            #features = scaler.transform(df)
            #features = pd.DataFrame(features, columns=X_columns)
            #features = features.loc[:, mask]
            #inactive_prob, active_prob = model.predict_proba(
            #    features.to_numpy().reshape(1, -1))[0]
            #activity, confidence = True, active_prob
            #if inactive_prob > active_prob:
            #    activity, confidence = False, inactive_prob
            #results = {}
            #results["smiles"] = form.smiles.data
            #results["model"] = full_names_dict[form.model.data]
            #results["type_of_activity"] = full_names_dict[form.class_to_pred.data]
            #results["activity"] = activity
            #results["confidence"] = confidence

            #flash("Please scroll to the bottom of the page to see the results", "success")
            #return render_template('pred.html', form=form, results=results)

        except pd.errors.EmptyDataError:
            error = "Please make sure the SMILES are valid and try again"
            #flash(error, "danger")
            subprocess.run(f"rm ./{desc_filename}", shell=True)


        return jsonify(prediction)

api.add_resource(typhoidClassifier, '/salutem')

if __name__ == '__main__':
    # Load model
    # with open('modules/model.pickle', 'rb') as f:
    #    //sav.load()
    #    model = pickle.load(f)

    #with open('modules/Results/LR.sav', 'rb') as lr:
    #   lr_model = pickle.load(lr)

    app.run(debug=True)