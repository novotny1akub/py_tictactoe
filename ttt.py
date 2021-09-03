# https://learnxinyminutes.com/docs/cs-cz/python/
# https://piskvorky.jobs.cz/api/doc

# predelat place_move_global a place_move_and_return do jedne procedury
# změnit print do angličtiny
# nazvy procedur (REST_check_status, GAME_func), aby vyjadrovali to, co procedura skutecne dela
# REST_user aby zapisovala do txt user_token

import numpy as np
import pandas as pd
import requests
import json
import time
from datetime import datetime
import multiprocessing
from joblib import Parallel, delayed
import os

os.chdir('C:/Users/novotny/Desktop/covid/py_ttt/')

with open('user_token.txt','r') as f:
    user_token = f.read()

def create_empty_board(ncols=28, nrows=25):
    cols = np.arange(-ncols, ncols + 1, 1)
    rows = np.arange(-nrows, nrows + 1, 1)
    
    global board
    
    board = pd.DataFrame({
        "col": np.tile(cols, len(rows)), # np.tile repeats the array n times
        "row": np.repeat(rows, len(cols)), # np.repeat repeats each value of the array ntimes
        "value": "_"
        })
    board = board.pivot(index="row", columns="col", values="value")
    board = board.sort_index(ascending=False)

def place_move_global(row, col, mark):
    global board
    board.at[row, col] = mark
    
def place_move_and_return(board, row, col, mark):
    board = board.copy(deep=True)
    board.at[row, col] = mark
    return board

def place_best_move(mark, game_token, user_token):
    global board
    best_move_df = best_move(board, mark=mark)
    print(best_move_df)
    place_move_global(best_move_df.rows, best_move_df.cols, mark=mark)
    REST_play(int(best_move_df.rows), int(best_move_df.cols), game_token=game_token, user_token=user_token) # přidat token

def board_wide_to_long(board):
    df = board.copy(deep=True)
    cols = df.columns
    df['rows'] = df.index
    df = pd.melt(df, id_vars="rows", value_vars=cols, var_name="cols")
    return df

def best_move(board, mark="x"):
    df = board.copy(deep=True)
    df = board_wide_to_long(board)
    df = df[df.value != "_"]
    
    max_row = max(df.rows) + 1
    min_row = min(df.rows) - 1
    max_col = max(df.cols) + 1
    min_col = min(df.cols) - 1
    
    df_move_candidates = board_wide_to_long(board)
    df_move_candidates = df_move_candidates[df_move_candidates.value == "_"]
    df_move_candidates = df_move_candidates.query("rows <= @max_row & rows >= @min_row & cols <= @max_col & cols >= @min_col").reset_index(drop=True)
    df_move_candidates["value"] = mark

    # go through all move candidates, and "play" them, later evaluate
    boards = []
    for index, row in df_move_candidates.iterrows():
        boards.append(place_move_and_return(board, row['rows'], row['cols'], row['value']))
        
    df_move_candidates["boards"] = boards
    veval_board = np.vectorize(eval_board)
    df_move_candidates["score"] = veval_board(df_move_candidates.boards, mark)
    del df_move_candidates["boards"]
    df_move_candidates = df_move_candidates.sort_values(by='score', ascending=False)
    df_move_candidates = df_move_candidates.loc[df_move_candidates.score == df_move_candidates.score.max()].sample(n=1)
    df_move_candidates = df_move_candidates[["rows", "cols"]]

    return df_move_candidates

def eval_board(board, mark="x"):
    # extract all rows, columns, ascending diagonals, descending diagonals
    # filter those containing only _
    # assign scores to r, c, ad, dd
    df = board.copy(deep=True)
    df = board_wide_to_long(df)
    df['dd'] = df.cols - df.rows + 54 # descending diagonal
    df['ad'] = df.rows + df.cols + 54 # ascending diagonal
    
    strs = pd.concat([
        df.groupby('rows')['value'].agg(lambda col: ''.join(col)).reset_index(drop=True),
        df.groupby('cols')['value'].agg(lambda col: ''.join(col)).reset_index(drop=True),
        df.groupby('dd')['value'].agg(lambda col: ''.join(col)).reset_index(drop=True),
        df.groupby('ad')['value'].agg(lambda col: ''.join(col)).reset_index(drop=True)
        ])
    
    strs = strs[strs.str.contains('x|o', regex=True)]
    
    scores = strs.apply(eval_string, mark=mark).sum()
    
    return scores

def eval_string(strng, mark="x"):
    
    score = 0
    if mark == "x":
        a = "x" # attacking marks for the evaluation
        d = "o" # defending marks
    else:
        d = "x"
        a = "o"
    
    # kladne body
    if "___{a}___".format(a=a) in strng: score += 5e0
    elif "__{a}__".format(a=a) in strng: score += (5e0 - 1e0)
        
    if "__{a}{a}__".format(a=a) in strng: score += 5e1
    elif "_{a}{a}__".format(a=a) in strng or "__{a}{a}_".format(a=a) in strng or "__{a}_{a}__".format(a=a) in strng: score += (5e1 - 2e1)
    
    if "__{a}{a}{a}__".format(a=a) in strng or "_{a}{a}{a}__".format(a=a) in strng or "__{a}{a}{a}_".format(a=a) in strng: score += 5e3
    elif "_{a}_{a}{a}_".format(a=a) in strng or "_{a}{a}_{a}_".format(a=a) in strng: score += (5e3 - 1e3)
    
    if "_{a}{a}{a}{a}_".format(a=a) in strng: score += 5e5
    elif "{d}{a}{a}{a}{a}_".format(a=a, d=d) in strng or "_{a}{a}{a}{a}{d}".format(a=a, d=d) in strng: score += (2*5e3 - 3e3)
    elif "{d}{a}_{a}{a}{a}{d}".format(a=a, d=d) in strng or "{d}{a}{a}_{a}{a}{d}".format(a=a, d=d) in strng or "{d}{a}{a}{a}_{a}{d}".format(a=a, d=d) in strng: score += (2*5e3 - 3e3)
    
    if "{a}{a}{a}{a}{a}".format(a=a) in strng: score += 5e9
    
    # zaporne body
    if "{a}{d}{d}{d}__".format(a=a, d=d) in strng or "__{d}{d}{d}{a}".format(a=a, d=d) in strng: score -= 10e0 + 4e0
    elif "___{d}___".format(d=d) in strng or "__{d}__{d}__".format(d=d) in strng: score -= 10e0
    elif "__{d}__".format(d=d) in strng: score -= 10e0 - 1e0
    
    if "__{d}{d}__".format(d=d) in strng: score -= 10e1
    elif "_{d}{d}__".format(d=d) in strng or "__{d}{d}_".format(d=d) in strng or "__{d}_{d}__".format(d=d) in strng: score -= 10e1 - 2e1
    
    if "__{d}{d}{d}_".format(d=d) in strng or "_{d}{d}{d}__".format(d=d) in strng or "_{d}_{d}_{d}_".format(d=d) in strng or "_{d}{d}_{d}_".format(d=d) in strng or "_{d}_{d}{d}_".format(d=d) in strng: score -= 10e3

    if "{d}{d}{d}{d}_".format(d=d) in strng or "_{d}{d}{d}{d}".format(d=d) in strng or "{d}_{d}{d}{d}".format(d=d) in strng or "{d}{d}_{d}{d}".format(d=d) in strng or "{d}{d}{d}_{d}".format(d=d) in strng: score -= 10e5
    
    if "{d}{d}{d}{d}{d}".format(d=d) in strng or "_{d}{d}{d}{d}_".format(d=d) in strng: score -= 10e9
    
    return score

def if_board_empty_play(mark):
    global board

    df = board.copy(deep=True)
    df = board_wide_to_long(df)
    df = df[df.value != "_"]
    
    if df.empty: place_move_global(row=0, col=0, mark=mark)

def REST_user(nickname = "allGoodNamesAreGone", email = "novotnyjakub@email.cz"):
    
    json_in = {
        "nickname": nickname,
        "email": email
        }
    
    r = requests.post("https://piskvorky.jobs.cz/api/v1/user", json=json_in)
    
    return r.text

def REST_connect(user_token):
    
    json_in = {"userToken": user_token}
    
    r = requests.post("https://piskvorky.jobs.cz/api/v1/connect", json=json_in)
    
    return json.loads(r.text).get('gameToken')

def REST_check_status(game_token, user_token):
    
    json_in = {
        "userToken": user_token,
        "gameToken": game_token
        }
    
    r = requests.post("https://piskvorky.jobs.cz/api/v1/checkStatus", json=json_in)
    
    dct = json.loads(r.text)
    cross_p = 'ja' if dct.get('playerCrossId') == dct.get('actualPlayerId') else 'souper' # zacina
    circle_p = 'ja' if dct.get('playerCircleId') == dct.get('actualPlayerId') else 'souper' # druhy
    zacinam = cross_p == 'ja'
    
    if dct.get('playerCircleId') == None:
        print('Ceka se na protihrace - za 30s zkusim znovu')
        time.sleep(30)
        REST_check_status(game_token, user_token)
    elif r.status_code == 226:
        print('Hra uz skoncila')
        print(dct.get('winnerId'))
    elif zacinam and not dct.get('coordinates') and dct.get('playerCircleId') != None:
        REST_play(0, 0, game_token, user_token)
        if_board_empty_play(mark = "x")
        print("Zahran prvni obligatni tah 0, 0")
        REST_check_status(game_token, user_token)
    elif dct.get('coordinates'):
        df = pd.DataFrame(dct.get('coordinates'))
        df['playerId'] = df['playerId'].map(lambda x: 'ja' if x == dct.get('actualPlayerId') else 'souper')
        df['order_turn'] = (df.playerId == cross_p).map({True: 1, False: 2})
        df = df.rename(columns={"y": "row", "x": "column"}).iloc[::-1]
        df['turn_nr'] = np.maximum(
            (df.playerId == cross_p).cumsum(),
            (df.playerId == circle_p).cumsum()
            )
        df['mark'] = df['playerId'].map({'ja': 'x', 'souper': 'o'})
        df = df.sort_values(by=['turn_nr', 'order_turn']).reset_index(drop=True)
        df = df[['playerId', 'row', 'column', 'order_turn', 'turn_nr', 'mark']]
        
        # znovu vytvářím board dle toho, co mají na serveru
        create_empty_board()
        for index, row in df.iterrows():
            place_move_global(row['row'], row['column'], row['mark'])
        
        # pokud hral posledni souper, zahraju ja, jinak pockam 5s a zkusim status znovu
        if (df.playerId[-1::]== 'souper').bool():
            df = df.tail(1)
            place_move_global(df.row, df.column, mark="o")
            place_best_move(mark="x", game_token=game_token, user_token=user_token)
            REST_check_status(game_token, user_token)
        else:
            print('Cekam na souperuv tah')
            time.sleep(5)
            REST_check_status(game_token, user_token)
    else:
        print('Unexpected')
        
def REST_play(row, col, game_token, user_token):
    
    json = {
        "userToken": user_token,
        "gameToken": game_token,
        "positionX": col,
        "positionY": row
        }
    
    r = requests.post("https://piskvorky.jobs.cz/api/v1/play", json=json)
    
    return r.status_code

def GAME_func():
    create_empty_board()
    game_token = REST_connect(user_token=user_token)
    # print(game_token)
    REST_check_status(game_token=game_token, user_token=user_token)

def GAME_multi(times=2):
    num_cores = multiprocessing.cpu_count()
    runs_nr =  range(int((num_cores/2)*times))
    
    print(times, ":", datetime.now())
    Parallel(n_jobs=num_cores)(delayed(GAME_func)() for i in runs_nr)
    print(datetime.now())

def REVISED_game_func(game_token, user_token):
    create_empty_board()
    REST_check_status(game_token=game_token, user_token=user_token)

def REVISED_game_multi(user_token, games=2):
    # request n game_tokens
    # wait three minutes
    # process the games in bulk
    
    print(datetime.now(), "following nr of API calls planned", games)
    lst_game_tokens = list(map(lambda x: REST_connect(user_token=user_token), range(games)))
    print(datetime.now(), "API calls finished, waiting for 180s")
    time.sleep(180)
    # parallel processing
    print(datetime.now(), "proceeding with the ttt gs")
    num_cores = multiprocessing.cpu_count()    
    Parallel(n_jobs=num_cores)(delayed(REVISED_game_func)(game_token=tkn, user_token=user_token) for tkn in lst_game_tokens)
    print(datetime.now(), "finished now")
    
# board.loc[7:-7,-7:7]
# REVISED_game_multi(user_token, games=32)
"""
for i in range(4):
    REVISED_game_multi(user_token, games=32)
"""
