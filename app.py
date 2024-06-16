
model = keras.models.load_model("trained_model.h5")


target_size = (300, 300)

class_labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

st.set_page_config(
    page_title="Garbage Classification",
    page_icon=":recycle:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Garbage Classification")

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    st.image(image, caption='Uploaded image', width=300)
    st.write('The predicted class is:', class_labels[predicted_class])
