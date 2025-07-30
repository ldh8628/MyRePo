from flask import Flask, render_template, request, redirect, url_for, session, flash
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # 운영 시 환경변수로 관리하세요

# Jinja2에 intcomma 필터 등록
def intcomma(value):
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return value

app.jinja_env.filters['intcomma'] = intcomma

# 테스트용 사용자 데이터 (실제 DB로 대체 가능)
users = {
    'testuser': {
        'password': 'testpass',
        'balance': 1000,
        'history': []  # 거래 내역 저장
    },
    'seconduser': {
        'password': 'secondpass',
        'balance': 500,
        'history': []  # 거래 내역 저장
    }
}

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        userid = request.form['userid']
        pwd = request.form['password']
        user = users.get(userid)
        if user and user['password'] == pwd:
            session['userid'] = userid
            flash('로그인되었습니다.')
            return redirect(url_for('dashboard'))
        else:
            flash('아이디 또는 비밀번호가 올바르지 않습니다.')
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'userid' not in session:
        return redirect(url_for('login'))
    userid = session['userid']
    user = users[userid]
    history = user['history']  # 사용자별 persistent 거래 내역

    if request.method == 'POST':
        action = request.form['action']
        amount = float(request.form['amount'])
        deposit_amt = 0.0
        withdraw_amt = 0.0
        if action == 'deposit':
            user['balance'] += amount
            deposit_amt = amount
            flash(f'{intcomma(amount)}원 입금되었습니다.')
        elif action == 'withdraw':
            if amount <= user['balance']:
                user['balance'] -= amount
                withdraw_amt = amount
                flash(f'{intcomma(amount)}원 출금되었습니다.')
            else:
                flash('잔액이 부족합니다.')
        # 거래 내역에 추가
        record = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'deposit': deposit_amt,
            'withdraw': withdraw_amt,
            'balance': user['balance']
        }
        history.append(record)
        return redirect(url_for('dashboard'))

    return render_template('dashboard.html', userid=userid, balance=user['balance'], history=history)

@app.route('/logout')
def logout():
    session.pop('userid', None)
    flash('로그아웃되었습니다.')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)