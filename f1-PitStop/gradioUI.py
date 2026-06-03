import gradio as gr
from predict import predict_single


def predict_interface(lap_number, lap_time, tyre_compound, gap_to_leader, laps_on_tyre):
    res = predict_single(
        lap_number=int(lap_number),
        lap_time=float(lap_time),
        tyre_compound=tyre_compound,
        gap_to_leader=float(gap_to_leader),
        laps_on_tyre=int(laps_on_tyre)
    )
    # display nicely
    return {
        'pit_probability': f"{res['pit_probability']:.4f}",
        'will_pit_next': str(res['will_pit_next']),
        'threshold_used': res.get('threshold', 0.5)
    }


def build_ui():
    iface = gr.Interface(
        fn=predict_interface,
        inputs=[
            gr.Number(label='Lap number', value=20),
            gr.Number(label='Lap time (s)', value=85.3),
            gr.Dropdown(['soft', 'medium', 'hard'], label='Tyre compound', value='medium'),
            gr.Number(label='Gap to leader (s)', value=12.4),
            gr.Number(label='Laps on tyre', value=6)
        ],
        outputs=[gr.JSON(label='Prediction')],
        title='F1 Pit Stop Predictor',
        description='Enter lap-level features to predict probability of pitting next lap.'
    )
    return iface


if __name__ == '__main__':
    ui = build_ui()
    ui.launch(server_name='0.0.0.0', server_port=7860)
