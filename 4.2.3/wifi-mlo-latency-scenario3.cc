/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Latency Scenario #3  (based on [15] – Kozioł et al., accepted work)
 *
 * Purpose: evaluate how varying link configurations and MLO modes (STR/EMLSR)
 * affect DL latency.
 *
 * Single BSS: 1 AP + 1 STA, 1-3 links in STR or EMLSR mode.
 *   numLinks = 1 (SLO):  5 GHz "channel 1"   80 MHz
 *   numLinks = 2 (STR):  5 GHz "channel 1" + "channel 100", both 80 MHz
 *   numLinks = 3 (STR):  5 GHz "channel 1" + "channel 100" + "channel 136", all 80 MHz
 *   (per Table 3.9, test with STR and EMLSR modes)
 *
 * PHY:    EHT 802.11be, MCS 8, 2 SS, GI 800 ns, 80 MHz per link
 * A-MPDU: nMpdus = 1024
 * Traffic: constant-rate On-Off DL (AP → STA), fixed offered load swept via runner
 * AP–STA distance: 5 m
 * MLO mode: STR (default) or EMLSR (if emlsr=true)
 *
 * Output lines parsed by runner:
 *   "Mean DL Latency: X.XXX ms"
 *   "DL Latency p1: X.XXX ms"
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

NS_LOG_COMPONENT_DEFINE("WifiMloLatencyScenario3");

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

// ------} // namespace

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
    uint32_t    numLinks     = 2;     // 1, 2, or 3
    double      offeredLoad  = 500.0; // Mb/s; swept by runner
    bool        emlsr        = false; // false = STR, true = EMLSR

    CommandLine cmd(__FILE__);
    cmd.AddValue("simTime",     "Simulation time [s]",                   simTime);
    cmd.AddValue("payloadSize", "UDP payload size [B]",                  payloadSize);
    cmd.AddValue("nMpdus",      "Max MPDUs per A-MPDU",                  nMpdus);
    cmd.AddValue("numLinks",    "Number of links (1, 2, or 3)",          numLinks);
    cmd.AddValue("offeredLoad", "DL offered load [Mb/s]",                offeredLoad);
    cmd.AddValue("emlsr",       "Enable EMLSR mode (false=STR, true=EMLSR)", emlsr);
    cmd.Parse(argc, argv);

    if (numLinks < 1 || numLinks > 3)
        NS_ABORT_MSG("numLinks must be 1, 2, or 3");
    if (offeredLoad <= 0.0)
        NS_ABORT_MSG("offeredLoad must be > 0");

    RngSeedManager::SetSeed(1); // run number controlled by --RngRun

    const char* mloMode = emlsr ? "EMLSR" : "STR";
    std::cout << "=== Latency Scenario #3 ===\n"
              << "numLinks: " << numLinks
              << "  MLO mode: " << mloMode
              << "  offeredLoad: " << offeredLoad << " Mb/s\n\n";

    // -----------------------------------------------------------------------
    //  Channel plan (5 GHz only, 80 MHz per link):
    //  ns-3 ChannelSettings expects valid 80 MHz center channels in 5 GHz.
    //    Link 0: center ch42  (maps to "channel 1" block)
    //    Link 1: center ch106 (maps to "channel 100" block)  -- numLinks >= 2
    //    Link 2: center ch138 (maps to "channel 136" block)  -- numLinks == 3
    // -----------------------------------------------------------------------
    struct LinkCfg
    {
        uint32_t chanNum;
        uint32_t width; // MHz
        const char* band;
    };

    const LinkCfg linkCfgs[3] = {
        {42,  80, "BAND_5GHZ"},
        {106, 80, "BAND_5GHZ"},
        {138, 80, "BAND_5GHZ"},
    };

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
        phy.AddChannel(specChans[l], WIFI_SPECTRUM_5_GHZ);
        phy.Set(l, "ChannelSettings",             StringValue(oss.str()));
        phy.Set(l, "Antennas",                    UintegerValue(2));
        phy.Set(l, "MaxSupportedTxSpatialStreams", UintegerValue(2));
        phy.Set(l, "MaxSupportedRxSpatialStreams", UintegerValue(2));
    }

    // -----------------------------------------------------------------------
    //  WiFi (EHT 802.11be, MCS 8)
    // -----------------------------------------------------------------------
    // 256-QAM 3/4 (thesis Table 3.9 "MCS 8") -> EHT MCS 8 in 802.11be indexing.
    const char* DATA_MODE = "EhtMcs8";
    const char* CTL_MODE  = "EhtMcs0";

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211be);
    wifi.ConfigHeOptions("GuardInterval", TimeValue(NanoSeconds(800)));
    if (emlsr)
    {
        wifi.ConfigEhtOptions("EmlsrActivated",     BooleanValue(true));
        wifi.ConfigEhtOptions("TransitionTimeout",  TimeValue(MicroSeconds(1024)));
        wifi.ConfigEhtOptions("MediumSyncDuration", TimeValue(MicroSeconds(0)));
        wifi.ConfigEhtOptions("MsdOfdmEdThreshold", IntegerValue(-72));
        wifi.ConfigEhtOptions("MsdMaxNTxops",       UintegerValue(0));
    }
    for (uint32_t l = 0; l < numLinks; ++l)
        wifi.SetRemoteStationManager(l, "ns3::ConstantRateWifiManager",
                                     "DataMode",    StringValue(DATA_MODE),
                                     "ControlMode", StringValue(CTL_MODE));

    // -----------------------------------------------------------------------
    //  MAC helper  (no MU scheduler — single STA, SU transmission only)
    // -----------------------------------------------------------------------
    WifiMacHelper wifiMac;
    Ssid ssid = Ssid("mlo-lat3");

    wifiMac.SetType("ns3::ApWifiMac",
                    "Ssid",            SsidValue(ssid),
                    "BE_MaxAmpduSize", UintegerValue(nMpdus * (payloadSize + 60)),
                    "MpduBufferSize",  UintegerValue(nMpdus));
    NetDeviceContainer apDev = wifi.Install(phy, wifiMac, apNode);

    // EMLSR manager must be installed on the *non-AP* MLD before its devices
    // are created; without it, EmlsrActivated only advertises support and the
    // device behaves identically to STR. The EmlsrLinkSet enumerates which
    // setup links should run EMLSR (paper [15] uses all of them).
    if (emlsr && numLinks > 1)
    {
        std::ostringstream linkSet;
        for (uint32_t l = 0; l < numLinks; ++l)
        {
            if (l > 0) linkSet << ",";
            linkSet << l;
        }
        wifiMac.SetEmlsrManager("ns3::AdvancedEmlsrManager",
                                "EmlsrLinkSet",         StringValue(linkSet.str()),
                                "SwitchAuxPhy",         BooleanValue(true),
                                "PutAuxPhyToSleep",     BooleanValue(false),
                                "AuxPhyChannelWidth",   UintegerValue(80),
                                "AuxPhyMaxModClass",    StringValue("EHT"),
                                "AuxPhyTxCapable",      BooleanValue(true));
    }

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
    pos->Add(Vector(5.0, 0.0, 0.0)); // STA -- 5 m link distance per thesis Table 3.9
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
    // -----------------------------------------------------------------------
    const double tStart = 1.0;
    const double tStop  = simTime + 1.0;

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
    const double p1LatencyMs   = ChannelAccessPercentileMs(1.0);
    const double p99LatencyMs  = ChannelAccessPercentileMs(99.0);
    const double meanThrMbps   = (simTime > 0.0)
                                 ? (g_rxBytes * 8.0) / (simTime * 1e6)
                                 : 0.0;

    std::cout << "\nMean DL Latency: "   << meanLatencyMs << " ms\n";
    std::cout << "DL Latency p1: "      << p1LatencyMs    << " ms\n";
    std::cout << "DL Latency p99: "     << p99LatencyMs  << " ms\n";
    std::cout << "Mean DL Throughput: " << meanThrMbps   << " Mb/s\n";

    Simulator::Destroy();
    return 0;
}
