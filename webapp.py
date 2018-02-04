# Please use:
#    curl http://127.0.0.1:5000/getCategory -X POST -d "text=我是一篇文章"
# to see just the JSON result, and use
#    curl http://127.0.0.1:5000/getCategory -X POST -v -d "text=我是一篇文章"
# to see all the details.  Remember the return format is JSON

from keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
from keras.metrics import top_k_categorical_accuracy
from keras.models import load_model
from collections import Counter
import thulac
import pickle
import numpy as np

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)

# Get input
parser = reqparse.RequestParser()
parser.add_argument('text')

class GetCategory(Resource):

    def __init__(self):
        # Start returning data

        # Load existing model
        with open('tokenizer.pickle', 'rb') as handle:
            self.load_tokenizer = pickle.load(handle)

        with open('max_words.pickle', 'rb') as handle:
            self.load_max_words = pickle.load(handle)

        with open('categories.pickle', 'rb') as handle:
            self.categories = pickle.load(handle)

        # Split the sentense 
        self.thu = thulac.thulac(seg_only=True)
        
    def clean_and_cut(self, text):
        soup = BeautifulSoup(text, 'html.parser')
        new_text = soup.get_text()

        new_text_cut = self.thu.cut(new_text, text=True).replace("\n", "")
        return new_text_cut

    def post(self):
        def top_1_accuracy(y_true, y_pred):
            return top_k_categorical_accuracy(y_true, y_pred, k=1)

        args = parser.parse_args()
        
        if(args['text'].strip() == ""):
            abort(404, message="Input is empty")
        
        new_text = self.clean_and_cut(args['text'])
        new_X = self.load_tokenizer.texts_to_matrix(new_text)
        new_X = pad_sequences(new_X, maxlen=self.load_max_words, padding='post')

        model = load_model('text_full_model.h5', custom_objects={'top_1_accuracy': top_1_accuracy})
        predict_class = model.predict_classes(new_X)
        cls_counter = Counter(predict_class)

        # Return category
        idx = cls_counter.most_common(1)[0][0]
        final_category = np.sort(list(self.categories))[2]
        return {"category": final_category}, 200

api.add_resource(GetCategory, '/getCategory')
    
# Run App
if __name__ == '__main__':
    app.run(debug=True)