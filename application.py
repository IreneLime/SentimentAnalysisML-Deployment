from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask import (
    Flask,
    render_template,
    session,
    redirect,
    url_for,
)


# Load model to make predictions
def load_model(user_input):
    loaded_model = None
    with open("basic_classifier.pkl", "rb") as fid:
        loaded_model = pickle.load(fid)

    vectorizer = None
    with open("count_vectorizer.pkl", "rb") as vd:
        vectorizer = pickle.load(vd)

    # How to use model to predict
    prediction = loaded_model.predict(vectorizer.transform([user_input]))[0]
    return prediction


# User Info submission
class UserForm(FlaskForm):

    input = StringField("Enter News:", validators=[DataRequired()])
    submit = SubmitField("Submit")


# create and initialize a new Flask app
application = Flask(__name__)
application.config["SECRET_KEY"] = "secretkeyveryhardtoguessece444"
# load the config
application.config.from_object(__name__)
Bootstrap(application)


@application.route("/", methods=["GET", "POST"])
def index():
    # Prompt the user to submit a string of news
    form = UserForm()
    if form.validate_on_submit():
        old_value = session.get("input")
        if old_value != None and old_value != form.input.data:
            # Use model to predict the user's input data
            pred = load_model(form.input.data)
            if pred == "REAL":
                session["message"] = "This is real news."
            elif pred == "FAKE":
                session["message"] = "This is fake news."
            else:
                session["message"] = "Unknown news type."
        else:
            session["message"] = None

        session["input"] = form.input.data
        return redirect(url_for("index"))

    return render_template(
        "index.html",
        form=form,
        user_input=session.get("input"),
        output=session.get("message"),
    )


if __name__ == "__main__":
    application.run()
