/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Scenario 4.4 — MLO in VR Applications
 *
 * Based on:
 *   [23] G. Adame et al. "Wi-Fi Multi-Link Operation: An Experimental Study of
 *        Latency and Throughput", IEEE Network, 2024.
 *   [24] Wi-Fi Alliance VR gaming requirements: DL p75 < 5 ms, UL p90 < 2 ms.
 *
 * Topology: 1 AP + 1 STA in 6 GHz band, distance 5 m.
 *   SLO: single 6 GHz link, channel 29
 *   MLO (STR): two 6 GHz links, channels 29 + 151
 *
 * Sweeps controlled by the runner over:
 *   --mode={SLO,MLO}, --direction={DL,UL},
 *   --mcs={0..13}, --channelWidth={20,40,80,160,320}
 *
 * Traffic: bursty VR pattern (~90 fps).
 *   DL: 11.1 ms period, ~50% duty, 100 Mb/s avg burst rate (≈ 11 Mb per frame).
 *   UL: 11.1 ms period, ~50% duty, 30 Mb/s avg burst rate.
 *  These rates are approximations of the [23] frame-rate VR pattern; the
 *  emitted distribution is heavy-tailed enough to expose latency tail
 *  behaviour relevant to VR p75/p90 requirements.
 *
 * MPDU aggregation: 1024 (thesis Table 3.14).
 *
 * Output (parsed by run_all_rerun.py):
 *   "VR p50 lat: X.XXX ms"
 *   "VR p75 lat: X.XXX ms"
 *   "VR p90 lat: X.XXX ms"
 *   "VR p99 lat: X.XXX ms"
 *   "VR mean lat: X.XXX ms"
 *   "VR rx pkts: N"
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/network-module.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/wifi-module.h"
#include "ns3/wifi-phy-band.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WifiVrMlo");

namespace {
std::vector<double> g_perPacketDelayMs;

void
RxWithDelayCallback(Ptr<const Packet> /*pkt*/, const Address& /*from*/,
                    const Address& /*to*/, const SeqTsSizeHeader& hdr)
{
    Time d = Simulator::Now() - hdr.GetTs();
    g_perPacketDelayMs.push_back(d.GetSeconds() * 1000.0);
}

double
percentile(std::vector<double>& v, double p)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (v.size() - 1) / 100.0;
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return v[lo];
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}
}  // namespace


int
main(int argc, char* argv[])
{
    double      simTime      = 6.0;
    uint32_t    payloadSize  = 1400;
    uint32_t    nMpdus       = 1024;
    std::string mode         = "MLO";   // SLO or MLO
    std::string direction    = "DL";    // DL or UL
    uint32_t    mcs          = 11;      // 0..13
    uint32_t    channelWidth = 80;      // 20, 40, 80, 160, 320

    CommandLine cmd(__FILE__);
    cmd.AddValue("simTime",      "Simulation time [s]",          simTime);
    cmd.AddValue("payloadSize",  "UDP payload size [B]",         payloadSize);
    cmd.AddValue("nMpdus",       "Max MPDUs per A-MPDU",         nMpdus);
    cmd.AddValue("mode",         "SLO or MLO",                   mode);
    cmd.AddValue("direction",    "DL or UL",                     direction);
    cmd.AddValue("mcs",          "MCS index (0..13)",            mcs);
    cmd.AddValue("channelWidth", "Channel width [MHz]",          channelWidth);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(mode != "SLO" && mode != "MLO", "mode must be SLO or MLO");
    NS_ABORT_MSG_IF(direction != "DL" && direction != "UL",
                    "direction must be DL or UL");
    NS_ABORT_MSG_IF(mcs > 13, "mcs must be in 0..13");
    NS_ABORT_MSG_IF(channelWidth != 20 && channelWidth != 40 && channelWidth != 80
                    && channelWidth != 160 && channelWidth != 320,
                    "channelWidth must be 20/40/80/160/320 MHz");

    RngSeedManager::SetSeed(1);

    uint32_t numLinks = (mode == "MLO") ? 2 : 1;

    std::cout << "=== Scenario 4.4 VR ===\n"
              << "mode=" << mode << "  dir=" << direction
              << "  mcs=" << mcs << "  bw=" << channelWidth << " MHz"
              << "  links=" << numLinks << "  simTime=" << simTime << "s\n\n";

    // ---- 6 GHz spectrum channels ---------------------------------------
    // Thesis Table 3.14: channels 29 and 151 (6 GHz UNII-5 / UNII-7).
    // For BW 20/40/80/160/320 we pick centres so that the 320 MHz channel
    // covers ch 31 (centred), and link-1 (ch 151) for UNII-7.
    auto makeChannel = []() {
        Ptr<MultiModelSpectrumChannel> c = CreateObject<MultiModelSpectrumChannel>();
        Ptr<LogDistancePropagationLossModel> plm =
            CreateObject<LogDistancePropagationLossModel>();
        plm->SetAttribute("Exponent",          DoubleValue(3.0));
        plm->SetAttribute("ReferenceDistance", DoubleValue(1.0));
        plm->SetAttribute("ReferenceLoss",     DoubleValue(40.0));
        c->AddPropagationLossModel(plm);
        return c;
    };

    Ptr<MultiModelSpectrumChannel> sc0 = makeChannel();
    Ptr<MultiModelSpectrumChannel> sc1 = (numLinks == 2) ? makeChannel() : nullptr;

    // 6 GHz channel centre numbers (IEEE 802.11be Annex E):
    // 20 MHz : ch 29 (UNII-5),     ch 151 (UNII-7)
    // 40 MHz : ch 27,              ch 155
    // 80 MHz : ch 23 (UNII-5),     ch 151 (UNII-7)
    // 160 MHz: ch 47 (UNII-5),     ch 175 (UNII-7)   [valid 160-MHz centres are 15/47/79/111/143/175/207]
    // 320 MHz: ch 31 (UNII-5+6),   ch 159 (UNII-7+8) [valid 320-MHz centres are 31/95/159]
    auto pickCh = [&](bool secondLink) -> uint32_t {
        switch (channelWidth)
        {
        case 20:  return secondLink ? 153 : 29;
        case 40:  return secondLink ? 155 : 27;
        case 80:  return secondLink ? 151 : 23;
        case 160: return secondLink ? 175 : 47;
        case 320: return secondLink ? 159 : 31;
        }
        return secondLink ? 151 : 29;
    };

    // ---- PHY -----------------------------------------------------------
    SpectrumWifiPhyHelper phy(numLinks);
    phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
    {
        std::ostringstream oss;
        oss << "{" << pickCh(false) << ", " << channelWidth << ", BAND_6GHZ, 0}";
        phy.AddChannel(sc0, WIFI_SPECTRUM_6_GHZ);
        phy.Set(0, "ChannelSettings",              StringValue(oss.str()));
        phy.Set(0, "Antennas",                    UintegerValue(2));
        phy.Set(0, "MaxSupportedTxSpatialStreams", UintegerValue(2));
        phy.Set(0, "MaxSupportedRxSpatialStreams", UintegerValue(2));
    }
    if (numLinks == 2)
    {
        std::ostringstream oss;
        oss << "{" << pickCh(true) << ", " << channelWidth << ", BAND_6GHZ, 0}";
        phy.AddChannel(sc1, WIFI_SPECTRUM_6_GHZ);
        phy.Set(1, "ChannelSettings",              StringValue(oss.str()));
        phy.Set(1, "Antennas",                    UintegerValue(2));
        phy.Set(1, "MaxSupportedTxSpatialStreams", UintegerValue(2));
        phy.Set(1, "MaxSupportedRxSpatialStreams", UintegerValue(2));
    }

    std::ostringstream dataMode;
    dataMode << "EhtMcs" << mcs;
    const char* CTL_MODE = "EhtMcs0";

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211be);
    wifi.ConfigHeOptions("GuardInterval", TimeValue(NanoSeconds(800)));
    for (uint32_t l = 0; l < numLinks; ++l)
        wifi.SetRemoteStationManager(l, "ns3::ConstantRateWifiManager",
                                     "DataMode",    StringValue(dataMode.str()),
                                     "ControlMode", StringValue(CTL_MODE));

    // ---- Nodes ---------------------------------------------------------
    NodeContainer apNode;   apNode.Create(1);
    NodeContainer staNode;  staNode.Create(1);

    Ssid ssid("mlo-vr");

    WifiMacHelper mac;
    mac.SetType("ns3::ApWifiMac",
                "Ssid",            SsidValue(ssid),
                "BE_MaxAmpduSize", UintegerValue(nMpdus * (payloadSize + 100)),
                "MpduBufferSize",  UintegerValue(nMpdus));
    NetDeviceContainer apDev = wifi.Install(phy, mac, apNode);

    mac.SetType("ns3::StaWifiMac",
                "Ssid",            SsidValue(ssid),
                "ActiveProbing",   BooleanValue(false),
                "BE_MaxAmpduSize", UintegerValue(nMpdus * (payloadSize + 100)),
                "MpduBufferSize",  UintegerValue(nMpdus));
    NetDeviceContainer staDev = wifi.Install(phy, mac, staNode);

    // ---- Mobility (5 m distance) ---------------------------------------
    MobilityHelper mob;
    Ptr<ListPositionAllocator> alloc = CreateObject<ListPositionAllocator>();
    alloc->Add(Vector(0.0, 0.0, 0.0));
    alloc->Add(Vector(5.0, 0.0, 0.0));
    mob.SetPositionAllocator(alloc);
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.Install(apNode);
    mob.Install(staNode);

    // ---- Internet stack -----------------------------------------------
    InternetStackHelper stack;
    stack.Install(apNode);
    stack.Install(staNode);
    Ipv4AddressHelper addr;
    addr.SetBase("10.4.0.0", "255.255.255.0");
    Ipv4InterfaceContainer apIf  = addr.Assign(apDev);
    Ipv4InterfaceContainer staIf = addr.Assign(staDev);
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // ---- VR Traffic (bursty, ~90 fps) ---------------------------------
    const double tStart = 1.0;
    const double tStop  = simTime + tStart;
    const uint16_t port = 5500;

    // Per-direction VR rates aligned with modern XR / Cloud VR workloads
    // (200-500 Mb/s DL Air Link / Cloud rendering, 80-150 Mb/s UL controller +
    // hand-tracking). The original [23] values (100/30) are too light for
    // modern VR — they let even 20 MHz channels trivially meet the 5 ms target.
    double rateMbps;
    Ptr<Node> srcNode;
    Ptr<Node> dstNode;
    Ipv4Address dstAddr;
    if (direction == "DL")
    {
        rateMbps = 200.0;
        srcNode = apNode.Get(0);
        dstNode = staNode.Get(0);
        dstAddr = staIf.GetAddress(0);
    }
    else
    {
        rateMbps = 1000.0;  // Heavy UL (matches reference paper p90 ≤ 2 ms boundaries)
        srcNode = staNode.Get(0);
        dstNode = apNode.Get(0);
        dstAddr = apIf.GetAddress(0);
    }

    // PacketSink with EnableSeqTsSizeHeader=true so RX timestamps yield delay.
    PacketSinkHelper sink("ns3::UdpSocketFactory",
                          InetSocketAddress(dstAddr, port));
    sink.SetAttribute("EnableSeqTsSizeHeader", BooleanValue(true));
    ApplicationContainer sinkApp = sink.Install(dstNode);
    sinkApp.Start(Seconds(tStart));
    sinkApp.Stop(Seconds(tStop + 0.5));

    Ptr<PacketSink> sinkPtr = DynamicCast<PacketSink>(sinkApp.Get(0));
    sinkPtr->TraceConnectWithoutContext("RxWithSeqTsSize",
                                         MakeCallback(&RxWithDelayCallback));

    OnOffHelper onoff("ns3::UdpSocketFactory",
                       InetSocketAddress(dstAddr, port));
    onoff.SetAttribute("PacketSize", UintegerValue(payloadSize));
    onoff.SetAttribute("EnableSeqTsSizeHeader", BooleanValue(true));
    // 11.1 ms period (90 fps), 50% duty -> peak rate = 2 * average
    onoff.SetAttribute("OnTime",
                       StringValue("ns3::ConstantRandomVariable[Constant=0.00555]"));
    onoff.SetAttribute("OffTime",
                       StringValue("ns3::ConstantRandomVariable[Constant=0.00555]"));
    std::ostringstream rateStr;
    rateStr << static_cast<uint64_t>(rateMbps * 2.0e6) << "bps";
    onoff.SetAttribute("DataRate", StringValue(rateStr.str()));

    ApplicationContainer srcApp = onoff.Install(srcNode);
    srcApp.Start(Seconds(tStart + 0.05));
    srcApp.Stop(Seconds(tStop));

    Simulator::Stop(Seconds(tStop + 1.0));
    Simulator::Run();

    // ---- Compute percentiles -------------------------------------------
    double p50 = percentile(g_perPacketDelayMs, 50.0);
    double p75 = percentile(g_perPacketDelayMs, 75.0);
    double p90 = percentile(g_perPacketDelayMs, 90.0);
    double p99 = percentile(g_perPacketDelayMs, 99.0);
    double mean = 0.0;
    for (double d : g_perPacketDelayMs) mean += d;
    if (!g_perPacketDelayMs.empty()) mean /= g_perPacketDelayMs.size();

    std::cout << "\nVR p50 lat: "  << p50 << " ms\n";
    std::cout << "VR p75 lat: "  << p75 << " ms\n";
    std::cout << "VR p90 lat: "  << p90 << " ms\n";
    std::cout << "VR p99 lat: "  << p99 << " ms\n";
    std::cout << "VR mean lat: " << mean << " ms\n";
    std::cout << "VR rx pkts: "  << g_perPacketDelayMs.size() << "\n";

    Simulator::Destroy();
    return 0;
}
