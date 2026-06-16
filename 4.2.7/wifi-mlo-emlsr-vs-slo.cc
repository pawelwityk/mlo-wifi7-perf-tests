/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Latency & Throughput Scenario — EMLSR vs SLO under varying load
 *
 * Compares EMLSR multi-link operation against SLO (single-link) as
 * the offered DL load is swept from light to saturation.
 *
 * Single BSS: 1 AP + nStas STAs
 *   SLO:   1 link,  5 GHz ch42  80 MHz
 *   EMLSR: 2 links, 5 GHz ch42 + 6 GHz ch7  (both 80 MHz)
 *
 * PHY: EHT 802.11be, MCS 11, 2 SS, GI 800 ns, 80 MHz per link
 * A-MPDU: 1024 MPDUs
 * Traffic: normalized DL offered load (AP → each STA), split equally
 * AP–STA distance: 5 m
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

NS_LOG_COMPONENT_DEFINE("WifiMloEmlsrVsSlo");

// === Channel-access-delay instrumentation ===
namespace
{
std::unordered_map<uint64_t, Time> g_macTxTimes;
std::vector<double> g_channelAccessDelaysSec;

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

int
main(int argc, char* argv[])
{
    // -----------------------------------------------------------------------
    //  Parameters
    // -----------------------------------------------------------------------
    double      simTime        = 10.0;
    uint32_t    payloadSize    = 1500;
    uint32_t    nMpdus         = 1024;
    std::string mloMode        = "EMLSR"; // SLO | EMLSR
    uint32_t    nStas          = 1;       // 1, 4, 10
    double      normalizedLoad = 1.0;     // 0..1

    CommandLine cmd(__FILE__);
    cmd.AddValue("simTime",        "Simulation time [s]",              simTime);
    cmd.AddValue("payloadSize",    "UDP payload size [B]",             payloadSize);
    cmd.AddValue("nMpdus",         "Max MPDUs per A-MPDU",             nMpdus);
    cmd.AddValue("mloMode",        "SLO or EMLSR",                     mloMode);
    cmd.AddValue("nStas",          "Number of stations (1, 4, 10)",    nStas);
    cmd.AddValue("normalizedLoad", "Normalized offered load [0..1]",   normalizedLoad);
    cmd.Parse(argc, argv);

    if (mloMode != "SLO" && mloMode != "EMLSR")
        NS_ABORT_MSG("mloMode must be SLO or EMLSR");
    if (nStas < 1)
        NS_ABORT_MSG("nStas must be >= 1");
    if (normalizedLoad < 0.0 || normalizedLoad > 1.0)
        NS_ABORT_MSG("normalizedLoad must be in [0, 1]");

    const bool isEmlsr    = (mloMode == "EMLSR");
    const uint32_t nLinks = isEmlsr ? 2 : 1;

    RngSeedManager::SetSeed(1);

    std::cout << "=== EMLSR vs SLO ===\n"
              << "mode: " << mloMode
              << "  nStas: " << nStas
              << "  normalizedLoad: " << normalizedLoad
              << "  links: " << nLinks << "\n\n";

    // -----------------------------------------------------------------------
    //  Spectrum channels (residential propagation γ=3.5)
    // -----------------------------------------------------------------------
    auto makeChannel = []() {
        Ptr<MultiModelSpectrumChannel> c = CreateObject<MultiModelSpectrumChannel>();
        Ptr<LogDistancePropagationLossModel> plm =
            CreateObject<LogDistancePropagationLossModel>();
        plm->SetAttribute("Exponent", DoubleValue(3.5));
        plm->SetAttribute("ReferenceDistance", DoubleValue(1.0));
        plm->SetAttribute("ReferenceLoss", DoubleValue(40.0));
        c->AddPropagationLossModel(plm);
        return c;
    };

    Ptr<MultiModelSpectrumChannel> specCh5 = makeChannel();
    Ptr<MultiModelSpectrumChannel> specCh6;
    if (isEmlsr)
        specCh6 = makeChannel();

    // -----------------------------------------------------------------------
    //  Nodes
    // -----------------------------------------------------------------------
    NodeContainer apNode;   apNode.Create(1);
    NodeContainer staNodes; staNodes.Create(nStas);

    // -----------------------------------------------------------------------
    //  PHY
    // -----------------------------------------------------------------------
    SpectrumWifiPhyHelper phy(nLinks);
    phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
    if (nLinks > 1)
        phy.SetPcapCaptureType(WifiPhyHelper::PcapCaptureType::PCAP_PER_LINK);

    phy.AddChannel(specCh5, WIFI_SPECTRUM_5_GHZ);
    phy.Set(0, "ChannelSettings",             StringValue("{42, 80, BAND_5GHZ, 0}"));
    phy.Set(0, "Antennas",                    UintegerValue(2));
    phy.Set(0, "MaxSupportedTxSpatialStreams", UintegerValue(2));
    phy.Set(0, "MaxSupportedRxSpatialStreams", UintegerValue(2));

    if (isEmlsr)
    {
        phy.AddChannel(specCh6, WIFI_SPECTRUM_6_GHZ);
        phy.Set(1, "ChannelSettings",             StringValue("{7, 80, BAND_6GHZ, 0}"));
        phy.Set(1, "Antennas",                    UintegerValue(2));
        phy.Set(1, "MaxSupportedTxSpatialStreams", UintegerValue(2));
        phy.Set(1, "MaxSupportedRxSpatialStreams", UintegerValue(2));
    }

    // -----------------------------------------------------------------------
    //  WiFi (EHT MCS 11, 2SS, GI 800 ns)
    // -----------------------------------------------------------------------
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211be);
    wifi.ConfigHeOptions("GuardInterval", TimeValue(NanoSeconds(800)));
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",    StringValue("EhtMcs11"),
                                 "ControlMode", StringValue("EhtMcs0"));

    if (isEmlsr)
    {
        wifi.ConfigEhtOptions("EmlsrActivated",    BooleanValue(true));
        wifi.ConfigEhtOptions("TransitionTimeout", TimeValue(MicroSeconds(128)));
    }

    // -----------------------------------------------------------------------
    //  MAC
    // -----------------------------------------------------------------------
    WifiMacHelper wifiMac;
    Ssid ssid = Ssid("emlsr-lat");

    wifiMac.SetType("ns3::ApWifiMac",
                    "Ssid",            SsidValue(ssid),
                    "BE_MaxAmpduSize", UintegerValue(nMpdus * (payloadSize + 60)),
                    "MpduBufferSize",  UintegerValue(nMpdus));
    NetDeviceContainer apDev = wifi.Install(phy, wifiMac, apNode);

    if (isEmlsr)
    {
        wifiMac.SetEmlsrManager("ns3::DefaultEmlsrManager",
                                "EmlsrLinkSet",       StringValue("0,1"),
                                "SwitchAuxPhy",       BooleanValue(true),
                                "AuxPhyChannelWidth", UintegerValue(80),
                                "AuxPhyMaxModClass",  StringValue("EHT"),
                                "AuxPhyTxCapable",    BooleanValue(true),
                                "PutAuxPhyToSleep",   BooleanValue(false));
    }

    wifiMac.SetType("ns3::StaWifiMac",
                    "Ssid",            SsidValue(ssid),
                    "ActiveProbing",   BooleanValue(false),
                    "BE_MaxAmpduSize", UintegerValue(nMpdus * (payloadSize + 60)),
                    "MpduBufferSize",  UintegerValue(nMpdus));
    NetDeviceContainer staDev = wifi.Install(phy, wifiMac, staNodes);

    // -----------------------------------------------------------------------
    //  Mobility (AP at origin, STAs on circle r=5 m)
    // -----------------------------------------------------------------------
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> pos = CreateObject<ListPositionAllocator>();
    pos->Add(Vector(0.0, 0.0, 0.0));
    const double R = 5.0;
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
    //  Internet stack
    // -----------------------------------------------------------------------
    InternetStackHelper stack;
    stack.Install(apNode);
    stack.Install(staNodes);

    Ipv4AddressHelper addr;
    addr.SetBase("10.1.0.0", "255.255.255.0");
    addr.Assign(apDev);
    Ipv4InterfaceContainer staIf = addr.Assign(staDev);

    // NOTE: Ipv4GlobalRoutingHelper::PopulateRoutingTables() is intentionally
    // omitted here. It calls WifiNetDevice::GetChannel() under the hood, which
    // ns-3 forbids on multi-link devices (EMLSR/STR). Single BSS (AP+STAs on
    // one /24) needs no IP routing — ARP + L2 handles everything.

    // -----------------------------------------------------------------------
    //  Traffic
    //  Reference peak (SLO, 80 MHz, 1 STA, nMpdus=1024): 1114 Mb/s
    // -----------------------------------------------------------------------
    const double tStart = 1.0;
    const double tStop  = simTime + 1.0;
    uint16_t     port   = 5000;

    const double singleLinkPeakMbps = 1114.0;
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
    //  Channel-access-delay traces
    // -----------------------------------------------------------------------
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx",
                    MakeCallback(&ChannelAccessMacTxTrace));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyTxBegin",
                    MakeCallback(&ChannelAccessPhyTxBeginTrace));

    // -----------------------------------------------------------------------
    //  Throughput via MacRx on STA side
    // -----------------------------------------------------------------------
    static uint64_t g_rxBytes = 0;
    Config::ConnectWithoutContext(
        "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacRx",
        MakeBoundCallback(+[](uint64_t* counter, Ptr<const Packet> p) {
            *counter += p->GetSize();
        }, &g_rxBytes));

    Simulator::Stop(Seconds(tStop + 1.0));
    Simulator::Run();

    // -----------------------------------------------------------------------
    //  Results
    // -----------------------------------------------------------------------
    double meanLatencyMs = ChannelAccessMeanMs();
    double meanThrMbps = (simTime > 0.0)
                         ? (g_rxBytes * 8.0) / (simTime * 1e6)
                         : 0.0;

    std::cout << "\nMean DL Latency: "    << meanLatencyMs << " ms\n";
    std::cout << "Mean DL Throughput: " << meanThrMbps   << " Mb/s\n";

    Simulator::Destroy();
    return 0;
}
