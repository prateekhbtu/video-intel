from fastapi import FastAPI, Request
app = FastAPI()

@app.post("/events")
async def events(req: Request):
    batch = await req.json()
    return {"ok": True, "n": len(batch)}
