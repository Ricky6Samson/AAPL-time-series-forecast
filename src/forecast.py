def multi_step_forecast(model,last_data,steps,features):
    import pandas as pd

    future_preds=[]
    current_input=last_data.copy()
    
    for _ in range(steps):
        input_df = pd.DataFrame([current_input[features]])
        pred=model.predict(input_df)[0]
        future_preds.append(pred)
    
        current_input['lag_3'] = current_input['lag_2']
        current_input['lag_2'] = current_input['lag_1']
        current_input['lag_1'] = pred
  
    return future_preds