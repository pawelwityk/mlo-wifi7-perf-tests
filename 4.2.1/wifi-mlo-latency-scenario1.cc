/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Latency Scenario #1  (based on [16] – Jeknic & Kocan, IEEE IT 2024)
 *
 * Analyses how different numbers of links and stations affect DL latency.
 *
 * Single BSS: 1 AP + nStas STAs
 *   1 link  → 5 GHz ch42  (80 MHz) or ch50  (160 MHz)
 *   2 links → 5 GHz ch42 + ch106 (80 MHz) or ch50 + ch114 (160 MHz)
 *   ("Channel 1" / "Channel 100" from the paper map to these center channels)
 *
 * PHY: EHT 802.11be, MCS 11, 2 SS, GI 800 ns
 * A-MPDU: 1024 MPDUs (required for proper 160 MHz operation in ns-3)
 * Traffic: normalized offered DL load (AP → each STA), equally split per user
 * AP–STA distance: STAs placed on a circle of radius 5 m
 *
 * Delay metric: channel access delay (MacTx → PhyTxBegin per packet UID)
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
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WifiMloLatencyScenario1");

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
    uint32_t    nMpdus       = 1024;  // A-MPDU size (large to avoid ns-3 queue
                                      // starvation at 160 MHz PHY rates)
    uint32_t    numLinks     = 2;    // 1 or 2
    uint32_t    channelWidth = 80;   // 80 or 160 MHz
    uint32_t    nStas        = 4;    // 1, 4, or 10
    double      normalizedLoad = 1.0; // 0..1, referenced to SLO peak for given width

    CommandLine cmd(__FILE__);
    cmd.AddValue("simTime",      "Simulation time [s]",           simTime);
    cmd.AddValue("payloadSize",  "UDP payload size [B]",          payloadSize);
    cmd.AddValue("nMpdus",       "Max MPDUs per A-MPDU",          nMpdus);
    cmd.AddValue("numLinks",     "Number of links (1 or 2)",      numLinks);
    cmd.AddValue("channelWidth", "Channel width per link [MHz]",  channelWidth);
    cmd.AddValue("nStas",        "Number of stations (1,4,10)",   nStas);
    cmd.AddValue("normalizedLoad", "Normalized offered load [0..1]", normalizedLoad);
    cmd.Parse(argc, argv);

    if (numLinks < 1 || numLinks > 2)
        NS_ABORT_MSG("numLinks must be 1 or 2");
    if (channelWidth != 80 && channelWidth != 160)
        NS_ABORT_MSG("channelWidth must be 80 or 160");
    if (nStas < 1)
        NS_ABORT_MSG("nStas must be >= 1");
    if (normalizedLoad < 0.0 || normalizedLoad > 1.0)
        NS_ABORT_MSG("normalizedLoad must be in [0, 1]");

    RngSeedManager::SetSeed(1); // run number controlled by --RngRun

    std::cout << "=== Latency Scenario #1 ===\n"
              << "numLinks: " << numLinks
              << "  channelWidth: " << channelWidth << " MHz"
              << "  nStas: " << nStas
              << "  normalizedLoad: " << normalizedLoad << "\n\n";

    // -----------------------------------------------------------------------
    //  Channel center numbers
    //  Paper "Channel 1"   → 80 MHz: ch42,  160 MHz: ch50
    //  Paper "Channel 100" → 80 MHz: ch106, 160 MHz: ch114
    // -----------------------------------------------------------------------
    uint32_t ch0 = (channelWidth == 80) ? 42 : 50;
    uint32_t ch1 = (channelWidth == 80) ? 106 : 114;

    // -----------------------------------------------------------------------
    //  Nodes
    // -----------------------------------------------------------------------
    NodeContainer apNode;   apNode.Create(1);
    NodeContainer staNodes; staNodes.Create(nStas);

    // -----------------------------------------------------------------------
    //  Spectrum channels (one per link — ensures cross-band isolation)
    // -----------------------------------------------------------------------
    // Residential building propagation environment (per Jeknic & Kocan, IT 2024).
    // Log-distance with exponent 3.5 approximates the residential model used in
    // the original paper (matlab residential propagation model has similar slope).
    auto makeResidentialChannel = []() {
        Ptr<MultiModelSpectrumChannel> c = CreateObject<MultiModelSpectrumChannel>();
        Ptr<LogDistancePropagationLossModel> plm =
            CreateObject<LogDistancePropagationLossModel>();
        plm->SetAttribute("Exponent", DoubleValue(3.5));
        plm->SetAttribute("ReferenceDistance", DoubleValue(1.0));
        plm->SetAttribute("ReferenceLoss", DoubleValue(40.0));
        c->AddPropagationLossModel(plm);
        return c;
    };

    Ptr<MultiModelSpectrumChannel> specCh0 = makeResidentialChannel();

    Ptr<MultiModelSpectrumChannel> specCh1;
    if (numLinks == 2)
    {
        specCh1 = makeResidentialChannel();
    }

    // -----------------------------------------------------------------------
    //  PHY helper
    // -----------------------------------------------------------------------
    SpectrumWifiPhyHelper phy(numLinks);
    phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
    if (numLinks > 1)
        phy.SetPcapCaptureType(WifiPhyHelper::PcapCaptureType::PCAP_PER_LINK);

    {
        std::ostringstream oss;
        oss << "{" << ch0 << ", " << channelWidth << ", BAND_5GHZ, 0}";
        phy.AddChannel(specCh0, WIFI_SPECTRUM_5_GHZ);
        phy.Set(0, "ChannelSettings",             StringValue(oss.str()));
        phy.Set(0, "Antennas",                    UintegerValue(2));
        phy.Set(0, "MaxSupportedTxSpatialStreams", UintegerValue(2));
        phy.Set(0, "MaxSupportedRxSpatialStreams", UintegerValue(2));
    }
    if (numLinks == 2)
    {
        std::ostringstream oss;
        oss << "{" << ch1 << ", " << channelWidth << ", BAND_5GHZ, 0}";
        phy.AddChannel(specCh1, WIFI_SPECTRUM_5_GHZ);
        phy.Set(1, "ChannelSettings",             StringValue(oss.str()));
        phy.Set(1, "Antennas",                    UintegerValue(2));
        phy.Set(1, "MaxSupportedTxSpatialStreams", UintegerValue(2));
        phy.Set(1, "MaxSupportedRxSpatialStreams", UintegerValue(2));
    }

    // -----------------------------------------------------------------------
    //  WiFi (EHT 802.11be, MCS 11)
    // -----------------------------------------------------------------------
    const char* DATA_MODE = "EhtMcs11";
    const char* CTL_MODE  = "EhtMcs0";

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211be);
    wifi.ConfigHeOptions("GuardInterval", TimeValue(NanoSeconds(800)));
    for (uint32_t l = 0; l < numLinks; ++l)
        wifi.SetRemoteStationManager(l, "ns3::ConstantRateWifiManager",
                                     "DataMode",    StringValue(DATA_MODE),
                                     "ControlMode", StringValue(CTL_MODE));

    // -----------------------------------------------------------------------
    //  MAC helper
    //  No multi-user scheduler: the latency scenario studies contention
    //  effects (queuing delay vs. #STAs) using SU round-robin scheduling,
    //  which is the model in [16].  DL OFDMA is not assumed.
    //  Using RrMultiUserScheduler with EHT and nStas >= 9 also triggers a
    //  known ns-3 bug (placeholder RuSpec shared by ≥9 candidates causes
    //  WifiTxVector::IsValid() to return false → SIGABRT).
    // -----------------------------------------------------------------------
    WifiMacHelper wifiMac;

    Ssid ssid = Ssid("mlo-lat1");

    // Install AP first
    wifiMac.SetType("ns3::ApWifiMac",
                    "Ssid",            SsidValue(ssid),
                    "BE_MaxAmpduSize", UintegerValue(nMpdus * (payloadSize + 60)),
                    "MpduBufferSize",  UintegerValue(nMpdus));
    NetDeviceContainer apDev = wifi.Install(phy, wifiMac, apNode);

    // Install STAs
    wifiMac.SetType("ns3::StaWifiMac",
                    "Ssid",          SsidValue(ssid),
                    "ActiveProbing", BooleanValue(false),
                    "BE_MaxAmpduSize", UintegerValue(nMpdus * (payloadSize + 60)),
                    "MpduBufferSize",  UintegerValue(nMpdus));
    NetDeviceContainer staDev = wifi.Install(phy, wifiMac, staNodes);

    // -----------------------------------------------------------------------
    //  Mobility
    //  AP at origin; STAs arranged on a circle of radius 5 m.
    // -----------------------------------------------------------------------
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> pos = CreateObject<ListPositionAllocator>();
    pos->Add(Vector(0.0, 0.0, 0.0)); // AP

    const double R = 5.0; // m
    for (uint32_t i = 0; i < nStas; ++i)
    {
        double angle = 2.0 * M_PI * i / nStas;
        pos->Add(Vector(R * std::cos(angle), R * std::sin(angle), 0.0));
    }
    mobility.SetPositionAllocator(pos);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(apNode);
    mobility.Install(staNodes);

    // -----------------------------------------------------------------------
    //  Internet stack + IP
    // -----------------------------------------------------------------------
    InternetStackHelper stack;
    stack.Install(apNode);
    stack.Install(staNodes);

    Ipv4AddressHelper addr;
    addr.SetBase("10.1.0.0", "255.255.255.0");
    Ipv4InterfaceContainer apIf  = addr.Assign(apDev);
    Ipv4InterfaceContainer staIf = addr.Assign(staDev);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // -----------------------------------------------------------------------
    //  Traffic: total offered load is normalized to single-link SLO peak,
    //  then split equally across STAs.
    //  Reference SLO peaks measured from 4.1.1 with nMpdus=1024:
    //    80 MHz  -> 1114 Mb/s
    //    160 MHz -> 1279 Mb/s  (mean of 20 runs; min=1122, max=1443)
    // -----------------------------------------------------------------------
    const double tStart = 1.0;
    const double tStop  = simTime + 1.0;
    uint16_t     port   = 5000;

    const double singleLinkPeakMbps = (channelWidth == 80) ? 1114.0 : 1279.0;
    const double totalOfferedLoadMbps = normalizedLoad * singleLinkPeakMbps;
    const double perStaLoadMbps = totalOfferedLoadMbps / static_cast<double>(nStas);

    for (uint32_t i = 0; i < nStas; ++i)
    {
        UdpServerHelper server(port);
        ApplicationContainer sApp = server.Install(staNodes.Get(i));
        sApp.Start(Seconds(tStart));
        sApp.Stop(Seconds(tStop));

        UdpClientHelper client(staIf.GetAddress(i), port);
        client.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
        if (perStaLoadMbps <= 0.0)
        {
            client.SetAttribute("Interval", TimeValue(Seconds(simTime + 10.0)));
        }
        else
        {
            const double pktBits = static_cast<double>(payloadSize) * 8.0;
            const double intervalUs = pktBits / perStaLoadMbps;
            client.SetAttribute("Interval", TimeValue(MicroSeconds(std::max(1.0, intervalUs))));
        }
        client.SetAttribute("PacketSize", UintegerValue(payloadSize));
        ApplicationContainer cApp = client.Install(apNode.Get(0));
        cApp.Start(Seconds(tStart));
        cApp.Stop(Seconds(tStop));
        ++port;
    }

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
    double meanLatencyMs = ChannelAccessMeanMs();
    double meanThrMbps = (simTime > 0.0)
                         ? (g_rxBytes * 8.0) / (simTime * 1e6)
                         : 0.0;

    // Always emit results; flag with SATURATED when delivered < 95 % of offered
    // so plotters can render those points with a distinct (hollow) marker.
    double deliveryRatio = (totalOfferedLoadMbps > 0.0 && simTime > 0.0)
                               ? (g_rxBytes * 8.0) / (totalOfferedLoadMbps * 1e6 * simTime)
                               : 1.0;
    if (deliveryRatio < 0.95)
    {
        std::cout << "SATURATED: delivery_ratio=" << deliveryRatio
                  << " offered=" << totalOfferedLoadMbps << " Mb/s"
                  << " delivered=" << meanThrMbps << " Mb/s\n";
    }

    std::cout << "\nMean DL Latency: "    << meanLatencyMs << " ms\n";
    std::cout << "Mean DL Throughput: " << meanThrMbps   << " Mb/s\n";

    Simulator::Destroy();
    return 0;
}
