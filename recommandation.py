from flask import Flask, request, jsonify, render_template
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

# Initialize Flask app
app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Recommendation Engine") \
    .getOrCreate()

# Load the dataset
data = spark.read.csv('C:/Users/JAIMIN/Downloads/ml-100k/ml-100k/u.data', sep='\t', header=False, inferSchema=True)
data = data.withColumnRenamed("_c0", "user_id") \
    .withColumnRenamed("_c1", "item_id") \
    .withColumnRenamed("_c2", "rating") \
    .withColumnRenamed("_c3", "timestamp")

# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2])

# Initialize and train ALS model
als = ALS(
    userCol='user_id',
    itemCol='item_id',
    ratingCol='rating',
    nonnegative=True,
    implicitPrefs=False,
    coldStartStrategy="drop"
)
model = als.fit(train_data)

@app.route('/')
def home():
    return render_template('index.html')

# Define route to get recommendations for a specific user
@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    # Get top 10 recommendations for the user
    user_recs = model.recommendForAllUsers(10)

    # Filter recommendations for the specific user
    recs = user_recs.filter(col("user_id") == user_id).collect()

    if not recs:
        return jsonify({"error": "No recommendations found for the given user ID."}), 404

    # Extract recommended item_ids and ratings
    recommendations = recs[0].recommendations

    # Return recommendations as JSON
    return jsonify({
        "user_id": user_id,
        "recommendations": [{"item_id": rec.item_id, "rating": rec.rating} for rec in recommendations]
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False)