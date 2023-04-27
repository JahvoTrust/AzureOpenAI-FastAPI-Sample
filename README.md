# AzureOpenAI-FastAPI-Sample
Azure OpenAI를 이용하여 파이썬 API를 작성하는 예제입니다. 
- Root에 .env 파일을 추가하여 환경변수를 만들고, Key등 필요한 정보를 설정합니다.(Key정보를 노출하지 않도록 주의하세요.)
```
OPENAI_API_KEY=.........................
OPENAI_API_BASE=https://<base-url>.openai.azure.com/
```
- API실행을 위해서 uvicorn 설치
```
pip install uvicorn
```
- pip install 를 이용하여 필요한 package들 설치합니다.
- .py 파일을 만든후 아래처럼 실행하면 API 실행됩니다.
```
uvicorn <file-name>:app --reload
```
- 브라우저 창에서 아래와 같이 입력하면 API명세를 확인할수 있습니다.
```
http://localhost:8000/docs
```
![Screenshot that shows the FastAPI Docs](media/docs.png "Overview of the APIs")
