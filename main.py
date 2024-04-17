import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

st.title('Online Chess game Analysis & Prediction Results')

url = 'Data_Cleaned.csv'
df = pd.read_csv(url)

st.subheader("Dataset")
st.write(df.head(5))
# Define categories
st.subheader("Produksi Buah")
fruit_columns = ['Oranges  Production (tonnes)', 'Bananas  Production ( tonnes)', 'Avocados Production (tonnes)', 'Apples Production (tonnes)', 'Grapes  Production (tonnes)', 'Tomatoes Production (tonnes)']
fruit_production_df = df.groupby('Year')[fruit_columns].sum().reset_index()
fruit_production_df['Total Production (tonnes)'] = fruit_production_df[fruit_columns].sum(axis=1)

for column in fruit_columns:
    fruit_production_df[f'{column.split("Production")[0].strip()} Production (%)'] = (fruit_production_df[column] / fruit_production_df['Total Production (tonnes)']) * 100

# Plotting
plt.figure(figsize=(10, 6))

for column in fruit_columns:
    plt.plot(fruit_production_df['Year'], fruit_production_df[column], marker='o', label=column.split('Production')[0].strip())

plt.title("Produksi Buah Tahunan")
plt.xlabel("Tahun")
plt.ylabel("Produksi ")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Simpan plot ke dalam suatu variabel
fruit_plot = plt.gcf()

# Tampilkan plot menggunakan st.pyplot()
st.pyplot(fruit_plot)

# Menampilkan persentase produksi dari total produksi makanan dan total produksi pertanian dunia
st.write("Persentase Produksi Buah Tahunan dari Total Produksi Pertanian Dunia:")
st.write(fruit_production_df[['Year'] + [f'{column.split("Production")[0].strip()} Production (%)' for column in fruit_columns]].round(2))

st.write("""
**Interpretasi:**
Tren produksi buah-buahan dari tahun 1961 hingga 2021 menunjukkan peningkatan yang konsisten, terutama untuk jeruk, pisang, apel, dan alpukat. Meskipun fluktuasi terjadi, produksi keseluruhan cenderung meningkat seiring waktu, dengan tomat dan anggur juga menunjukkan tren positif meskipun dengan variasi yang lebih besar.

**Actionable Insight:**
1. Terus Fokus pada Buah Unggulan: Petani dan industri pertanian harus terus memperhatikan produksi jeruk, pisang, apel, dan alpukat karena tren peningkatan ini. Perluasan lahan pertanian dan penggunaan teknologi modern dapat membantu meningkatkan produksi.
   
2. Analisis Fluktuasi Produksi: Penting untuk melakukan analisis mendalam terhadap faktor-faktor yang mempengaruhi fluktuasi produksi, seperti cuaca ekstrem, penyakit tanaman, atau perubahan permintaan pasar. Dengan pemahaman yang lebih baik, petani dapat mengurangi risiko fluktuasi tersebut.

3. Peluang Pasar: Dengan peningkatan produksi tomat dan anggur, pelaku industri perlu melakukan analisis pasar untuk memastikan keberlanjutan ekonomi dalam menghadapi persaingan yang mungkin ada. Inovasi dalam pengolahan dan pemasaran juga dapat membantu memanfaatkan peluang pasar yang ada.
""")

st.subheader("Produksi Pertanian")
agricultural_columns = ['Maize Production (tonnes)', 'Rice  Production ( tonnes)', 'Yams  Production (tonnes)', 'Wheat Production (tonnes)', 'Tea  Production ( tonnes )', 'Sweet potatoes  Production (tonnes)', 'Sunflower seed  Production (tonnes)', 'Sugar cane Production (tonnes)', 'Soybeans  Production (tonnes)', 'Rye  Production (tonnes)', 'Potatoes  Production (tonnes)', 'Peas, dry Production ( tonnes)', 'Palm oil  Production (tonnes)', 'Coffee, green Production ( tonnes)', 'Cocoa beans Production (tonnes)']

agricultural_production_df = df.groupby('Year').sum().reset_index()
agricultural_production_df['Total Production (tonnes)'] = agricultural_production_df[agricultural_columns].sum(axis=1)

# Menghitung persentase produksi
for column in agricultural_columns:
    agricultural_production_df[f'{column.split("Production")[0].strip()} Production (%)'] = (agricultural_production_df[column] / agricultural_production_df['Total Production (tonnes)']) * 100

# Plotting
num_subplots = 3
fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 16), sharex=True)

for i in range(num_subplots):
    start_index = i * (len(agricultural_columns) // num_subplots)
    end_index = min(start_index + (len(agricultural_columns) // num_subplots), len(agricultural_columns))

    for column in agricultural_columns[start_index:end_index]:
        axs[i].plot(agricultural_production_df['Year'], agricultural_production_df[column], marker='o', label=column.split('Production')[0].strip())

    axs[i].set_ylabel('Produksi ')
    axs[i].grid(True)
    axs[i].legend()

plt.title("Produksi Pertanian Tahunan")
plt.xlabel("Tahun")

plt.tight_layout()

# Menyimpan plot ke dalam variabel
agricultural_plot = plt.gcf()

# Menampilkan plot menggunakan st.pyplot()
st.pyplot(agricultural_plot)

# Menampilkan persentase produksi diluar visualisasi
st.write("Persentase Produksi Tahunan dari Total Produksi Pertanian Dunia:")
st.write(agricultural_production_df[['Year'] + [f'{column.split("Production")[0].strip()} Production (%)' for column in agricultural_columns]].round(2))

import streamlit as st

st.write("""
**Interpretasi:**
Tren produksi berbagai jenis tanaman pangan dari tahun 1961 hingga 2021 menunjukkan fluktuasi yang beragam. Produksi beras dan kedelai cenderung meningkat seiring waktu, sementara produksi gandum, kentang, dan kelapa sawit menunjukkan tren stabil atau sedikit meningkat. Sementara itu, produksi jagung, singkong, tebu, dan kacang kedelai memiliki fluktuasi yang lebih besar.

**Actionable Insight:**
1. Diversifikasi Tanaman: Petani dan industri pertanian perlu mempertimbangkan diversifikasi tanaman untuk mengurangi risiko fluktuasi produksi. Investasi dalam tanaman yang menunjukkan stabilitas produksi seperti beras dan kedelai dapat memberikan keuntungan jangka panjang.
   
2. Pengembangan Teknologi: Penerapan teknologi modern dalam pertanian, seperti irigasi yang efisien dan penggunaan varietas tanaman yang unggul, dapat membantu meningkatkan produktivitas dan stabilitas produksi.
   
3. Analisis Pasar: Pelaku industri perlu melakukan analisis pasar secara berkala untuk memahami permintaan konsumen dan tren pasar. Hal ini akan membantu dalam pengambilan keputusan strategis terkait penanaman dan pemasaran produk pertanian.
""")

st.write("Produksi Peternakan")
meat_columns = ['Meat, chicken  Production (tonnes)']

meat_production_df = df.groupby('Year').sum().reset_index()

# Plotting
plt.figure(figsize=(10, 6))

for column in meat_columns:
    plt.plot(meat_production_df['Year'], meat_production_df[column], marker='o', label=column.split('Production')[0].strip())

plt.title("Annual Chicken Meat Production")
plt.xlabel("Year")
plt.ylabel("Production (tonnes)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Saving the plot to a variable
meat_plot = plt.gcf()

# Displaying the plot using st.pyplot()
st.pyplot(meat_plot)

st.write("Annual Chicken Meat Production:")
st.write(meat_production_df[['Year'] + [column for column in meat_columns]].round(0))

import streamlit as st

st.write("""
**Interpretasi:**
Produksi daging ayam dari tahun 1961 hingga 2021 mengalami peningkatan yang signifikan, dengan beberapa fluktuasi pada periode tertentu. Terjadi peningkatan yang konsisten hingga tahun 1990-an, diikuti oleh pertumbuhan yang lebih lambat pada tahun 2000-an, namun masih menunjukkan tren positif secara keseluruhan.

**Actionable Insight:**
1. Peningkatan Konsumsi: Dari sisi konsumen, peningkatan produksi daging ayam dapat menawarkan lebih banyak pilihan dan ketersediaan produk. Masyarakat dapat memanfaatkan peningkatan ini untuk memperkaya pola makan mereka dengan sumber protein yang lebih sehat.

2. Peluang Bisnis: Pertumbuhan produksi daging ayam menciptakan peluang bisnis di sektor peternakan dan industri pengolahan makanan. Pelaku usaha dapat mempertimbangkan investasi dalam peternakan ayam atau pengembangan produk olahan daging ayam untuk memanfaatkan permintaan yang terus meningkat.

3. Keberlanjutan Lingkungan: Sementara meningkatnya produksi daging ayam dapat memberikan manfaat ekonomi, perlu juga diperhatikan dampaknya terhadap lingkungan. Pihak terkait, seperti petani dan produsen, harus memperhatikan praktik pertanian yang berkelanjutan untuk meminimalkan dampak negatif terhadap lingkungan.
""")

st.subheader("Model Prediksi")
file_path = 'gnb_model.pkl'
clf = joblib.load(file_path)

# Input form for agricultural production details
maize_production = st.number_input('Maize Production (tonnes)', value=0)  
rice_production = st.number_input('Rice Production (tonnes)', value=0)  
yams_production = st.number_input('Yams Production (tonnes)', value=0)  
wheat_production = st.number_input('Wheat Production (tonnes)', value=0)  
tomatoes_production = st.number_input('Tomatoes Production (tonnes)', value=0)  
tea_production = st.number_input('Tea Production (tonnes)', value=0)  
sweet_potatoes_production = st.number_input('Sweet potatoes Production (tonnes)', value=0)  
sunflower_seed_production = st.number_input('Sunflower seed Production (tonnes)', value=0)  
sugar_cane_production = st.number_input('Sugar cane Production (tonnes)', value=0)  
soybeans_production = st.number_input('Soybeans Production (tonnes)', value=0)  
rye_production = st.number_input('Rye Production (tonnes)', value=0)  
potatoes_production = st.number_input('Potatoes Production (tonnes)', value=0)  
oranges_production = st.number_input('Oranges Production (tonnes)', value=0)  
peas_dry_production = st.number_input('Peas, dry Production (tonnes)', value=0)  
palm_oil_production = st.number_input('Palm oil Production (tonnes)', value=0)  
grapes_production = st.number_input('Grapes Production (tonnes)', value=0)  
coffee_green_production = st.number_input('Coffee, green Production (tonnes)', value=0)  
cocoa_beans_production = st.number_input('Cocoa beans Production (tonnes)', value=0)  
meat_chicken_production = st.number_input('Meat, chicken Production (tonnes)', value=0)  
bananas_production = st.number_input('Bananas Production (tonnes)', value=0)  
avocados_production = st.number_input('Avocados Production (tonnes)', value=0)  
apples_production = st.number_input('Apples Production (tonnes)', value=0)  

# Prepare input data for prediction
input_data = [[maize_production, rice_production, yams_production, wheat_production, tomatoes_production,
               tea_production, sweet_potatoes_production, sunflower_seed_production, sugar_cane_production,
               soybeans_production, rye_production, potatoes_production, oranges_production,
               peas_dry_production, palm_oil_production, grapes_production, coffee_green_production,
               cocoa_beans_production, meat_chicken_production, bananas_production, avocados_production,
               apples_production]]

# Predict function
def predict_production(input_data):
    return clf.predict(input_data)

# Predict button
if st.button('Predict'):
    result = predict_production(input_data)
    # Display prediction result
    if result.size > 0:
        st.write('Hasil Prediksi Tahun Produksi:', result[0])
    else:
        st.write('Tidak dapat melakukan prediksi. Mohon periksa input Anda.')
