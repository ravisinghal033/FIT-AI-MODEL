@app.route('/start_low', methods=['POST'])
def start_low():
    global max_reps, start_video
    max_reps = 5
    start_video = True
    return '', 204
@app.route('/start_mid', methods=['POST'])
def start_mid():
    global max_reps, start_video
    max_reps = 10
    start_video = True
    return '', 204
@app.route('/start_high', methods=['POST'])
def start_high():
    global max_reps, start_video
    max_reps = 20
    start_video = True
    return '', 204