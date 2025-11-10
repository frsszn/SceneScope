import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import plotly.express as px
import pandas as pd
import time
from PIL import Image

# LOAD MODEL

@st.cache_resource
def load_model():
	model = tf.keras.models.load_model("src/best_model.keras")
	return model

model = load_model()


# KELAS

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Icon 
icons = {
	'buildings': '🏢',
	'forest': '🌳',
	'glacier': '❄️',
	'mountain': '🏔️',
	'sea': '🌊',
	'street': '🚗'
}



# BUAT TABS

tab1, tab2, tab3 = st.tabs(["🏠 Home", "📸 Prediksi Gambar", "ℹ️ Tentang"])


# TAB 1: HOME

with tab1:
	st.title("🌄 SceneScope: Landscape Image Classification App Demo")
	st.image("https://wallpapers.com/images/hd/1920x1080-full-hd-nature-landscape-54q8pleyp8lhhbu7.jpg")
	st.markdown(
		"""
		Selamat datang di **SceneScope**!

		Aplikasi ini menggunakan model **ResNet50 (Transfer Learning)**  
		untuk mengenali enam jenis pemandangan:
		- 🏢 Buildings  
		- 🌳 Forest  
		- ❄️ Glacier  
		- 🏔️ Mountain  
		- 🌊 Sea  
		- 🚗 Street  

		Upload gambar di tab **📸 Prediksi Gambar** untuk melihat hasil prediksi!
		"""
	)


# TAB 2: PREDIKSI GAMBAR

with tab2:
	st.header("📸 Upload dan Prediksi")
	uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

	if uploaded_file is not None:
		img = Image.open(uploaded_file).convert("RGB")
		st.image(img, caption="Gambar yang diupload", use_container_width=True)

		img_height, img_width = 224, 224
		img = img.resize((img_width, img_height))
		img_array = image.img_to_array(img)
		img_array = np.expand_dims(img_array, axis=0)
		img_array = preprocess_input(img_array)

		# Animasi loading
		st.write("⏳ Sedang memproses gambar...")
		with st.spinner('🔍 Menganalisis...'):
			progress = st.progress(0)
			for i in range(100):
				time.sleep(0.01)
				progress.progress(i + 1)
			predictions = model.predict(img_array)

		predicted_class = class_names[np.argmax(predictions)]
		confidence = np.max(predictions)

		st.success(f"🌟 Prediksi: {icons.get(predicted_class, '🖼️')} **{predicted_class.upper()}** ({confidence:.2%})")
       

		# Grafik probabilitas
		df = pd.DataFrame({
			"Kelas": class_names,
			"Probabilitas": predictions[0]
		})
		fig = px.bar(df, x="Kelas", y="Probabilitas", color="Kelas",
					 title="📈 Confidence Tiap Kelas", text="Probabilitas")
		fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
		st.plotly_chart(fig, use_container_width=True)


# TAB 3: TENTANG
with tab3:
	st.header("ℹ️ Tentang SceneScope")
	st.markdown(
		"""
		Aplikasi ini dibuat untuk mendemonstrasikan kemampuan **Transfer Learning neural network dengan ResNet50**
		dalam mengenali gambar pemandangan dari dataset **Intel Image Classification**.

		**ResNet50** digunakan karena:
		- Sudah dilatih dengan dataset besar (ImageNet), sehingga memiliki fitur visual yang kuat.  
		- Cocok untuk klasifikasi citra dengan berbagai pola kompleks seperti gunung, laut, dan hutan.  
		- Memberikan akurasi tinggi (93%) pada dataset ini dengan waktu pelatihan yang efisien.

		🚀 Dikembangkan dengan **Python**, **TensorFlow**, **Keras**, dan **Streamlit**.
		"""
	)

