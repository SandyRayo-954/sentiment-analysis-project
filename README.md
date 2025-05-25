# Sentiment Analysis GUI

This project is a Sentiment Analysis Tool with a graphical user interface (GUI) built using *Python* and *Tkinter*.  
It analyzes the sentiment of input text as Positive, Neutral, or Negative, and displays the confidence score.

## Features

- Real-time sentiment prediction  
- Displays confidence score of the prediction  
- Save results to a text file  
- User-friendly and simple GUI interface  
- Visual pie chart of sentiment results  

## How to Use

1. Ensure the dataset file realistic_sentiment140_500.csv is saved on your Desktop.
2. Run the sentiment_gui.py file using Python.
3. Enter the text you want to analyze into the input box.
4. Click the *Analyze* button to get the sentiment and confidence score.
5. Click the *Save Result* button to save the analyzed result to a file named sentiment_results.txt.
6. Click *Show Pie Chart* to visualize the distribution of analyzed sentiments.

## Requirements

- Python 3.x

### Required Python Packages

- pandas  
- nltk  
- scikit-learn  
- matplotlib  
- tkinter (usually included with Python)

You can install the required packages using:

```bash
pip install pandas nltk scikit-learn matplotlib
