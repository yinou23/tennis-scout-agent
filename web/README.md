# TSA Web, Week 5 Step 2

Next.js 前端，只做展示，不算任何数字。

运行前，先重启 API，因为 Step 2 给 api.py 加了 CORS，只允许 http://127.0.0.1:3000 和 http://localhost:3000 访问 http://127.0.0.1:8000，否则浏览器会因为跨域拦住请求。API 本身的 /health 和 /report 逻辑没有变。

1. 在 ~/TSA 里，把 scripts/api.py 换成 Step 2 的 api 文件，重启 API：
   source .venv/bin/activate
   set -a; source .env; set +a
   python scripts/api.py
2. 在另外一个终端，进 ~/TSA/web：
   cp .env.local.example .env.local
   npm install
   npm run dev
3. 打开 http://localhost:3000，先看 API 状态是不是已连接，再输入问题，点生成报告。
4. 成功才显示报告全文和来源与核查，失败只显示错误，不显示没过的报告。
5. 测完在 API 终端按 Ctrl-C，在前端终端按 Ctrl-C，分别停掉。

默认问题是 Jannik Sinner vs Carlos Alcaraz Hard，预期校准概率 0.6378，融合前 0.6698，核查 ok，规划器来源 llm。
