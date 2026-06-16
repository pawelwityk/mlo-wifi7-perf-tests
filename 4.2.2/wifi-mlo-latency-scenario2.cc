/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Latency Scenario #2  (based on [6] – Carrascosa-Zamacois et al., PIMRC 2023)
 *
 * Purpose: evaluate how different traffic loads affect DL latency.
 *
 * Single BSS: 1 AP + 1 STA, STR multi-link operation.
 *   numLinks = 1 (SLO): 5 GHz ch42  80 MHz
 *   numLinks = 2 (STR): 5 GHz ch42  + 6 GHz ch7,  both 80 MHz
 *   numLinks = 4 (STR): 5 GHz ch42  + 5 GHz ch106 + 6 GHz ch7 + 6 GHz ch23
 *                       (Channels "1" and "100" in each band, per Table 3.8)
 *
 * PHY:    EHT 802.11be, MCS 8, 1 SS, GI 800 ns, 80 MHz per link
 * A-MPDU: nMpdus = 1024
 * Traffic: constant-rate On-Off DL (AP → STA), offered load swept 100–2500 Mb/s
 * AP–STA distance: 5 m
 *
 * Output lines parsed by runner:
 *   "Mean DL Latency: X.XXX ms"
 *   "Mean DL Throughput: X.XX Mb/s"
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/network-module.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/wifi-module.h"
#include "ns3/wifi-phy-band.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WifiMloLatencyScenario2");

// === Channel-access-delay instrumentation ===
namespace
{
std::unordered_map<uint64_t, Time> g_macTxTimes;
std::vector<double> g_channelAccessDelaysSec;
uint64_t g_rxBytes = 0;

void
ChannelAccessMacTxTrace(std::string /*context*/, Ptr<const Packet> packet)
{
    g_macTxTimes[packet->GetUid()] = Simulator::Now();
}

void
ChannelAccessPhyTxBeginTrace(std::string /*context*/,
                             Ptr<const Packet> packet,
                             double /*txPowerW*/)
{
    auto it = g_macTxTimes.find(packet->GetUid());
    if (it == g_macTxTimes.end())
        return;
    g_channelAccessDelaysSec.push_back((Simulator::Now() - it->second).GetSeconds());
    g_macTxTimes.erase(it);
}

void
MacRxTrace(Ptr<const Packet> packet)
{
    g_rxBytes += packet->GetSize();
}

double
ChannelAccessMeanMs()
{
    if (g_channelAccessDelaysSec.empty())
        return 0.0;
    double s = std::accumulate(g_channelAccessDelaysSec.begin(),
                               g_channelAccessDelaysSec.end(), 0.0);
    return (s / static_cast<double>(g_channelAccessDelaysSec.size())) * 1000.0;
}

double
ChannelAccessPercentileMs(double p)
{
    if (g_channelAccessDelaysSec.empty())
        return 0.0;
    std::vector<double> sorted(g_channelAccessDelaysSec);
    std::sort(sorted.begin(), sorted.end());
    double rank = (p / 100.0) * static_cast<double>(sorted.size() - 1);
    size_t lower = static_cast<size_t>(rank);
    size_t upper = std::min(lower + 1, sorted.size() - 1);
    double frac = rank - static_cast<double>(lower);
    return (sorted[lower] + (sorted[upper] - sorted[lower]) * frac) * 1000.0;
}
} // namespace
// === end channel-access-delay instrumentation ===

// ---------------------------------------------------------------------------

int
main(int argc, char* argv[])
{
    // -----------------------------------------------------------------------
    //  Parameters
    // -----------------------------------------------------------------------
    double      simTime      = 10.0;
    uint32_t    payloadSize  = 1500;
    uint32_t    nMpdus       = 1024;
    uint32_t    numLinks     = 2;     // 1, 2, or 4
    double      offeredLoad  = 500.0; // Mb/s; swept by runner
    double      startupGuard = 2.0;   // seconds before traffic start
    uint32_t    maxMacQueuePackets = 4096;

    CommandLine cmd(__FILE__);
    cmd.AddValue("simTime",     "Simulation time [s]",                   simTime);
    cmd.AddValue("payloadSize", "UDP payload size [B]",                  payloadSize);
    cmd.AddValue("nMpdus",      "Max MPDUs per A-MPDU",                  nMpdus);
    cmd.AddValue("numLinks",    "Number of links (1, 2, or 4)",          numLinks);
    cmd.AddValue("offeredLoad", "DL offered load [Mb/s]",                offeredLoad);
    cmd.AddValue("startupGuard", "Idle time before application start [s]", startupGuard);
    cmd.AddValue("maxMacQueuePackets", "Max Wi-Fi MAC queue size [packets]", maxMacQueuePackets);
    cmd.Parse(argc, argv);

    if (numLinks != 1 && numLinks != 2 && numLinks != 4)
        NS_ABORT_MSG("numLinks must be 1, 2, or 4");
    if (offeredLoad <= 0.0)
        NS_ABORT_MSG("offeredLoad must be > 0");
    if (startupGuard < 0.0)
        NS_ABORT_MSG("startupGuard must be >= 0");
    if (maxMacQueuePackets == 0)
        NS_ABORT_MSG("maxMacQueuePackets must be > 0");

    RngSeedManager::SetSeed(1); // run number controlled by --RngRun

    // Bound buffering to keep overload behavior realistic and comparable.
    std::ostringstream qSize;
    qSize << maxMacQueuePackets << "p";
    Config::SetDefault("ns3::WifiMacQueue::MaxSize", QueueSizeValue(QueueSize(qSize.str())));

    std::cout << "=== Latency Scenario #2 ===\n"
              << "numLinks: " << numLinks
              << "  offeredLoad: " << offeredLoad << " Mb/s\n\n";

    // -----------------------------------------------------------------------
    //  Channel plan (80 MHz per link):
    //    Link 0: 5 GHz ch42   ("Channel 1"   in 5 GHz)
    //    Link 1: 6 GHz ch7    ("Channel 1"   in 6 GHz)   — numLinks >= 2
    //    Link 2: 5 GHz ch106  ("Channel 100" in 5 GHz)   — numLinks == 4
    //    Link 3: 6 GHz ch23   ("Channel 100" in 6 GHz)   — numLinks == 4
    // -----------------------------------------------------------------------
    struct LinkCfg
    {
        uint32_t              chanNum;
        uint32_t              width;   // MHz
        FrequencyRange        range;
        const char*           band;    // for ChannelSettings string
    };

    // clang-format off
    const LinkCfg linkCfgs[4] = {
        { 42,  80, WIFI_SPECTRUM_5_GHZ, "BAND_5GHZ" },
        {  7,  80, WIFI_SPECTRUM_6_GHZ, "BAND_6GHZ" },
        { 106, 80, WIFI_SPECTRUM_5_GHZ, "BAND_5GHZ" },
        { 23,  80, WIFI_SPECTRUM_6_GHZ, "BAND_6GHZ" },
    };
    // clang-format on

    // -----------------------------------------------------------------------
    //  Nodes
    // -----------------------------------------------------------------------
    NodeContainer apNode;  apNode.Create(1);
    NodeContainer staNode; staNode.Create(1);

    // -----------------------------------------------------------------------
    //  Spectrum channels — one independent object per link
    // -----------------------------------------------------------------------
    std::vector<Ptr<MultiModelSpectrumChannel>> specChans(numLinks);
    for (uint32_t l = 0; l < numLinks; ++l)
    {
        specChans[l] = CreateObject<MultiModelSpectrumChannel>();
        // Residential indoor propagation per thesis Table 3.1 (LogDistance gamma=3.5, refLoss=40 dB @ 1 m).
        Ptr<LogDistancePropagationLossModel> loss = CreateObject<LogDistancePropagationLossModel>();
        loss->SetAttribute("Exponent",          DoubleValue(3.5));
        loss->SetAttribute("ReferenceDistance", DoubleValue(1.0));
        loss->SetAttribute("ReferenceLoss",     DoubleValue(40.0));
        specChans[l]->AddPropagationLossModel(loss);
    }

    // -----------------------------------------------------------------------
    //  PHY helper
    // -----------------------------------------------------------------------
    SpectrumWifiPhyHelper phy(numLinks);
    phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
    if (numLinks > 1)
        phy.SetPcapCaptureType(WifiPhyHelper::PcapCaptureType::PCAP_PER_LINK);

    for (uint32_t l = 0; l < numLinks; ++l)
    {
        const LinkCfg& cfg = linkCfgs[l];
        std::ostringstream oss;
        oss << "{" << cfg.chanNum << ", " << cfg.width << ", " << cfg.band << ", 0}";
        phy.AddChannel(specChans[l], cfg.range);
        phy.Set(l, "ChannelSettings",             StringValue(oss.str()));
        // praca_mgr p.33 + Carrascosa [6] Sec III: two spatial streams.
        phy.Set(l, "Antennas",                    UintegerValue(2));
        phy.Set(l, "MaxSupportedTxSpatialStreams", UintegerValue(2));
        phy.Set(l, "MaxSupportedRxSpatialStreams", UintegerValue(2));
    }

    // -----------------------------------------------------------------------
    //  WiFi (EHT 802.11be, MCS 8)
    // -----------------------------------------------------------------------
    // 256-QAM 3/4 (thesis Table 3.8 "MCS 8") -> EHT MCS 8 in 802.11be indexing.
    const char* DATA_MODE = "EhtMcs8";
    const char* CTL_MODE  = "EhtMcs0";

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211be);
    wifi.ConfigHeOptions("GuardInterval", TimeValue(NanoSeconds(800)));
    for (uint32_t l = 0; l < numLinks; ++l)
        wifi.SetRemoteStationManager(l, "ns3::ConstantRateWifiManager",
                                     "DataMode",    StringValue(DATA_MODE),
                                     "ControlMode", StringValue(CTL_MODE));

    // -----------------------------------------------------------------------
    //  MAC helper  (no MU scheduler — single STA, SU transmission only)
    // -----------------------------------------------------------------------
    WifiMacHelper wifiMac;
    Ssid ssid = Ssid("mlo-lat2");

    wifiMac.SetType("ns3::ApWifiMac",
                    "Ssid",            SsidValue(ssid),
                    "BE_MaxAmpduSize", UintegerValue(nMpdus * (payloadSize + 60)),
                    "MpduBufferSize",  UintegerValue(nMpdus));
    NetDeviceContainer apDev = wifi.Install(phy, wifiMac, apNode);

    wifiMac.SetType("ns3::StaWifiMac",
                    "Ssid",            SsidValue(ssid),
                    "ActiveProbing",   BooleanValue(false),
                    "BE_MaxAmpduSize", UintegerValue(nMpdus * (payloadSize + 60)),
                    "MpduBufferSize",  UintegerValue(nMpdus));
    NetDeviceContainer staDev = wifi.Install(phy, wifiMac, staNode);

    // -----------------------------------------------------------------------
    //  Mobility: AP at origin, STA 5 m away
    // -----------------------------------------------------------------------
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> pos = CreateObject<ListPositionAllocator>();
    pos->Add(Vector(0.0, 0.0, 0.0)); // AP
    pos->Add(Vector(5.0, 0.0, 0.0)); // STA
    mobility.SetPositionAllocator(pos);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(apNode);
    mobility.Install(staNode);

    // -----------------------------------------------------------------------
    //  Internet stack + IP
    // -----------------------------------------------------------------------
    InternetStackHelper stack;
    stack.Install(apNode);
    stack.Install(staNode);

    Ipv4AddressHelper addr;
    addr.SetBase("10.1.0.0", "255.255.255.0");
    Ipv4InterfaceContainer apIf  = addr.Assign(apDev);
    Ipv4InterfaceContainer staIf = addr.Assign(staDev);
    // Global route population queries NetDevice::GetChannel(), which is not
    // valid for multi-link WifiNetDevice. This scenario is a single subnet
    // (AP <-> STA), so static host routing setup is unnecessary.

    // -----------------------------------------------------------------------
    //  Traffic: constant-rate On-Off DL (AP → STA)
    //  OnTime = always on, OffTime = 0  →  steady offered load = offeredLoad Mb/s.
    //  This models the MATLAB networkTrafficOnOff object from [6].
    // -----------------------------------------------------------------------
    const double tStart = 1.0 + startupGuard;
    const double tStop  = simTime + tStart;

    std::ostringstream drStr;
    drStr << static_cast<uint64_t>(offeredLoad * 1e6) << "bps";

    OnOffHelper onoff("ns3::UdpSocketFactory",
                      InetSocketAddress(staIf.GetAddress(0), 5000));
    onoff.SetAttribute("DataRate",   StringValue(drStr.str()));
    onoff.SetAttribute("PacketSize", UintegerValue(payloadSize));
    onoff.SetAttribute("OnTime",     StringValue("ns3::ConstantRandomVariable[Constant=1e9]"));
    onoff.SetAttribute("OffTime",    StringValue("ns3::ConstantRandomVariable[Constant=0]"));

    UdpServerHelper server(5000);
    ApplicationContainer sApp = server.Install(staNode.Get(0));
    sApp.Start(Seconds(tStart));
    sApp.Stop(Seconds(tStop));

    ApplicationContainer cApp = onoff.Install(apNode.Get(0));
    cApp.Start(Seconds(tStart));
    cApp.Stop(Seconds(tStop));

    // -----------------------------------------------------------------------
    //  Channel-access-delay traces + MacRx throughput
    // -----------------------------------------------------------------------
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx",
                    MakeCallback(&ChannelAccessMacTxTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyTxBegin",
                    MakeCallback(&ChannelAccessPhyTxBeginTrace));
    Config::ConnectWithoutContext(
        "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacRx",
        MakeCallback(&MacRxTrace));

    Simulator::Stop(Seconds(tStop + 1.0));
    Simulator::Run();

    // -----------------------------------------------------------------------
    //  Results
    // -----------------------------------------------------------------------
    const double meanLatencyMs = ChannelAccessMeanMs();
    const double p50LatencyMs  = ChannelAccessPercentileMs(50.0);
    const double p99LatencyMs  = ChannelAccessPercentileMs(99.0);
    const double meanThrMbps   = (simTime > 0.0)
                                 ? (g_rxBytes * 8.0) / (simTime * 1e6)
                                 : 0.0;

    std::cout << "\nMean DL Latency: "   << meanLatencyMs << " ms\n";
    std::cout << "DL Latency p50: "     << p50LatencyMs  << " ms\n";
    std::cout << "DL Latency p99: "     << p99LatencyMs  << " ms\n";
    std::cout << "Mean DL Throughput: " << meanThrMbps   << " Mb/s\n";

    Simulator::Destroy();
    return 0;
}
