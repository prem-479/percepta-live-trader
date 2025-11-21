import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import LiveSignals from "./pages/LiveSignals";
import News from "./pages/News";
import Risk from "./pages/Risk";
import Coach from "./pages/Coach";
import Portfolio from "./pages/Portfolio";

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <div style={{ padding: "20px", color: "white" }}>
        <Routes>
          <Route path="/" element={<LiveSignals />} />
          <Route path="/signals" element={<LiveSignals />} />
          <Route path="/news" element={<News />} />
          <Route path="/risk" element={<Risk />} />
          <Route path="/coach" element={<Coach />} />
          <Route path="/portfolio" element={<Portfolio />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
