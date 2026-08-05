# project_bank

Flask와 Jinja2로 로그인, 잔액 조회, 입금·출금과 거래내역 화면을 시연하는 인메모리 모의 은행 웹 애플리케이션이다.

## 실행

Python과 Flask가 필요하다.

> 이 앱은 `0.0.0.0`에 debug mode로 bind하고 고정 session secret과 평문 테스트 비밀번호를 사용한다. 격리된 로컬 데모 환경에서만 실행하고 외부 network에 노출하지 않는다.

```bash
cd project_bank
python3 app.py
```

브라우저에서 `http://localhost:5000`에 접속한다. 포함된 테스트 계정은 다음과 같다.

| 사용자 | 비밀번호 | 초기 잔액 |
| --- | --- | ---: |
| `testuser` | `testpass` | 1,000 |
| `seconduser` | `secondpass` | 500 |

입금·출금 form을 제출하면 현재 날짜, 입금액, 출금액과 변경된 잔액이 dashboard 거래내역에 추가된다.

## 출력과 제한

화면은 `templates/login.html`, `templates/dashboard.html`로 렌더링된다. 사용자·잔액·거래내역은 Python process memory에만 있으므로 server를 재시작하면 초기화된다.

이 코드는 교육용 데모다. CSRF 보호, database, 인증 보안과 동시성 처리가 없으므로 실제 금융·운영 환경에 사용하면 안 된다.
