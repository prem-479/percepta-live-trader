import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";

import Dashboard from "./pages/Dashboard";
import LiveSignals from "./pages/LiveSignals";
import News from "./pages/News";
import Risk from "./pages/Risk";
import Coach from "./pages/Coach";
import Portfolio from "./pages/Portfolio";

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/signals" element={<LiveSignals />} />
        <Route path="/news" element={<News />} />
        <Route path="/risk" element={<Risk />} />
        <Route path="/coach" element={<Coach />} />
        <Route path="/portfolio" element={<Portfolio />} />
      </Routes>
    </BrowserRouter>
  );
}
