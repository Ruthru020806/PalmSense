                preview = opened_preview.convert("RGB")

            st.html(
                '<div class="content-card">'
                '<div class="content-title">🖼️ Image preview</div>'
                '<div class="content-subtitle">'
                'Confirm this is the image you want to analyze.'
                '</div></div>'
            )
            st.image(preview, caption=uploaded_file.name, width="stretch")

            if st.button(
                "🔍 Analyze FFB image",
                width="stretch",
                key="prediction_analyze",
            ):
                success = run_prediction(uploaded_file)
                if success:
                    st.rerun()

        except Exception as error:
            st.error(
                "This file could not be opened or analyzed. "
                "Please choose a valid JPG, JPEG, or PNG image."
            )
            with st.expander("Technical error details"):
                st.exception(error)

    if st.session_state.current_result is not None:
        show_prediction_result()


def show_history():
    section_title("Prediction history", "Review results created during this app session.")
    if not st.session_state.history:
        st.html("""<div class="content-card">
          <div class="content-title">No predictions yet</div>
          <div class="content-subtitle">Your completed analyses will appear here. Open Prediction to upload your first image.</div>
        </div>""")
        if st.button("Go to prediction  →", width="stretch"):
            st.session_state.nav_page = "Prediction"
            st.rerun()
        return

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["Image", "Time", "Ripeness", "Confidence (%)", "Estimated count"])
    for item in st.session_state.history:
        writer.writerow([item["filename"], item["time"], item["ripeness"], f'{item["confidence"] * 100:.2f}', f'{item["count"]:.2f}'])
    st.download_button("⬇️ Download full session history (CSV)", csv_buffer.getvalue(), "palmsense_history.csv", "text/csv")

    for index, item in enumerate(st.session_state.history, start=1):
        st.html(f"""<div class="history-card">
          <div class="history-name">🌴 {index}. {html.escape(str(item['ripeness']))}</div>
          <div class="history-meta"><b>Image:</b> {html.escape(str(item['filename']))}<br>
          <b>Time:</b> {html.escape(str(item['time']))}<br>
          <b>Confidence:</b> {float(item['confidence']) * 100:.2f}%<br>
          <b>Estimated count:</b> {float(item['count']):.2f}</div>
        </div>""")


def show_about():
    section_title("About PalmSense", "A simple tool that checks an oil palm fruit-bunch image using your trained model.")
    st.html("""<div class="content-card">
      <div class="content-title">What the application provides</div>
      <p>• Four-class ripeness classification: Empty Bunch, Overripe, Ripe, and Underripe.</p>
      <p>• A model-generated mask visualization.</p>
      <p>• An estimated FFB count value.</p>
      <p>• Class probabilities and downloadable CSV summaries.</p>
    </div>
    <div class="content-card">
      <div class="content-title">Model workflow</div>
      <div class="step-row"><div class="step-num">1</div><div class="step-text"><strong>Input image</strong><span>Resize and normalize for model input.</span></div></div>
      <div class="step-row"><div class="step-num">2</div><div class="step-text"><strong>Shared feature extractor</strong><span>ConvNeXt-FPN feature fusion with Transformer attention.</span></div></div>
      <div class="step-row"><div class="step-num">3</div><div class="step-text"><strong>Multi-task heads</strong><span>Mask output, ripeness class, and count estimate.</span></div></div>
    </div>
    <div class="notice">Model limitation: the mask training targets are rectangular pseudo-masks derived from the dataset's existing bounding boxes, not hand-drawn polygon masks. The existing FFB dataset has one annotated bunch per image, so it does not establish performance for dense multi-bunch counting.</div>""")
    st.markdown("")
    m1, m2, m3 = st.columns(3)
    m1.metric("Validation accuracy", "89.19%")
    m2.metric("Test accuracy", "82.14%")
    m3.metric("Best checkpoint epoch", "10")
    st.caption("These figures are from the existing training/evaluation run and are not a guarantee of accuracy on every new image.")


# ------------------------------------------------------------
# Page router
# ------------------------------------------------------------
if current_page == "Home":
    show_home()
elif current_page == "Prediction":
    show_prediction_page()
elif current_page == "History":
    show_history()
else:
    show_about()

st.html("""
<div class="footer">
  <strong>PalmSense</strong> · Oil Palm Fruit Ripeness Prediction<br>
  Built around the existing multi-task deep learning model · Smart agriculture 🌱
</div>
""")
