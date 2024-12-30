### **1. Prerequisites**
- Python 3.8 or higher

---

### **2. Setup Instructions**

#### **2.1 Install Project Dependencies**
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/venkateshtata/gen-cortex
   cd gen-cortex
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### **3. Start the Server**
1. Run main:
   ```bash
   python main.py
   ```

2. The server will start at `http://localhost:8000`.

---

### **4. Test the API**
You can test the `/query` endpoint using the following `curl` command:

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{
          "query": "When is Ray Summit being held ?",
          "num_chunks": 5,
          "stream": true
        }'
```
