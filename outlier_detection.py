import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('soil moisture.csv')

sns.boxplot(x=df['Average Volumetric Soil Moisture (%)'])

# Show the plot
plt.savefig('soil_moisture_box_plot.png')
plt.close()
