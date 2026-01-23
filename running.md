# Running StructureCraft

## 1. Install Dependencies

```bash
pip install -r api/requirements.txt
```

## 2. Start the Backend

```bash
cd api
uvicorn main:app --reload --port 8000
```

## 3. Start the Frontend

```bash
cd web
npm install
npm run dev
```

## 4. Open in Browser

- **Frontend:** http://localhost:3000
- **API Docs:** http://localhost:8000/docs
