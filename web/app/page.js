"use client";
import { useEffect, useState } from "react";
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
export default function Home() {
  const [question, setQuestion] = useState("Jannik Sinner vs Carlos Alcaraz Hard");
  const [health, setHealth] = useState("检查中");
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((res) => res.json())
      .then((data) => setHealth(data.status === "ok" ? `已连接，${data.step}` : "API 返回异常"))
      .catch(() => setHealth("未连接，请先运行 python scripts/api.py"));
  }, []);
  async function onGenerate() {
    const q = question.trim();
    if (!q) {
      setError({ message: "问题不能为空" });
      setReport(null);
      return;
    }
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const res = await fetch(`${API_URL}/report`, {
        body: JSON.stringify({ question: q }),
        headers: { "Content-Type": "application/json" },
        method: "POST",
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail || data;
        setError({
          message: detail.error || "请求失败",
          plannerSource: detail.planner_source,
          plan: detail.plan,
          verification: detail.verification,
        });
        return;
      }
      setReport(data);
    } catch (e) {
      setError({ message: `网络错误，请确认 API 已启动在 ${API_URL}。${String(e)}` });
    } finally {
      setLoading(false);
    }
  }
  return (
    <main>
      <h1>Tennis Scout Agent</h1>
      <p className="note">报告里的每个数字都来自数据库查询，不是模型凭空写出。报告语言跟问题语言一致，中文问题出中文报告，英文问题出英文报告。数据截至 2026-05-25，不是实时数据。</p>
      <p className={health.includes("已连接") ? "health-ok" : "health-bad"}>API 状态：{health}</p>
      <label htmlFor="question">问题</label>
      <input id="question" type="text" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="例如：Jannik Sinner vs Carlos Alcaraz Hard" />
      <button onClick={onGenerate} disabled={loading}>{loading ? "生成中" : "生成报告"}</button>
      {error && (
        <section className="error">
          <h2>生成失败，没有报告</h2>
          <p>{error.message}</p>
          {error.plannerSource && <p>规划器来源：{error.plannerSource}</p>}
          {error.plan && <p>计划：{JSON.stringify(error.plan)}</p>}
          {error.verification && <p>核查：{JSON.stringify(error.verification)}</p>}
          <p>失败时不显示没过的报告，这是特意这样做的。</p>
        </section>
      )}
      {report && (
        <>
          <section className="card">
            <h2>报告</h2>
            <pre className="report">{report.final_text}</pre>
          </section>
          <section className="card">
            <h2>来源与核查</h2>
            <p>规划器来源：{report.planner_source}</p>
            {report.planner_source && report.planner_source !== "llm" && (
              <p>注意：本次用规则规划器兜底，原因：{report.planner_source}。大模型没有参与计划，答案数字仍来自工具。</p>
            )}
            <p>计划：{JSON.stringify(report.plan)}</p>
            <p>模型：{report.llm_model}</p>
            <p>图是否用了真 LangGraph：{report.graph_used_langgraph ? "是" : "否"}</p>
            <p>核查是否通过：{report.verification && report.verification.ok ? "是" : "否"}</p>
            {report.verification && <p>核查详情：{JSON.stringify(report.verification)}</p>}
            <p>数据截至 2026-05-25，不是实时数据。状态、局数、退赛、交手只做描述，不进概率。</p>
          </section>
        </>
      )}
    </main>
  );
}
