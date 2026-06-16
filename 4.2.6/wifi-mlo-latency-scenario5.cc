/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Latency Scenario #5  (Section 3.2.11 / Scenario 4.2.6)
 *
 * Based on [17]: Carrascosa-Zamacois et al.,
 *   "Wi-Fi multi-link operation: An experimental study of latency and throughput",
 *   IEEE/ACM Transactions on Networking 32.1 (2023), pp. 308–322.
 *
 * Goal: Assess how different channel occupancy levels influence the latency
 * of MLO-STR vs SLO, under symmetric and asymmetric OBSS interference.
 *
 * ── Topology ────────────────────────────────────────────────────────────────
 *   Main BSS : 1 AP + 1 STA, DL
 *              SLO  → single link on ch0 (the less-congested channel)
 *              MLO-STR → 2 links: ch0 + ch1
 *   OBSS-1   : 1 AP + numObssStas1 STAs, single-link on ch0  (always present
 *              when numObssStas1 > 0; occupies the primary channel)
 *   OBSS-2   : 1 AP + numObssStas2 STAs, single-link on ch1  (created only
 *              when numLinks==2 and numObssStas2 > 0)
 *
 * ── Channel plan (5 GHz, 80 MHz each) ──────────────────────────────────────
 *   ch0 : 5 GHz centre channel 42  (~5210 MHz  →  "ch36"  region in thesis)
 *   ch1 : 5 GHz centre channel 106 (~5530 MHz  →  "ch100" region in thesis)
 *
 * ── PHY / MAC ───────────────────────────────────────────────────────────────
 *   Standard  : 802.11be (EHT)
 *   Data mode : EhtMcs11,  2 SS,  GI 800 ns,  80 MHz
 *   A-MPDU    : nMpdus (default 1024) — Table 3.12
 *
 * ── Traffic ─────────────────────────────────────────────────────────────────
 *   Main BSS  : DL OnOff at offeredLoad Mb/s (Poisson arrivals, port 5000)
 *   OBSS-k    : DL OnOff per OBSS STA at obssRatePerSta Mb/s (port 5001/5002)
 *                 Default obssRatePerSta = 111.4 Mb/s  ≈ 10 % of single-link
 *                 capacity (1114 Mb/s from baseline Table 4.1).
 *                 1 STA → 10 %,  4 STAs → 40 %,  7 STAs → 70 %  occupancy.
 *
 * ── Output lines parsed by run_all_rerun.py ─────────────────────────────────
 *   "Mean DL Latency: X.XXX ms"
 *   "DL Latency p50: X.XXX ms"
 *   "DL Latency p95: X.XXX ms"
 *   "DL Latency p99: X.XXX ms"
 *   "Mean DL Throughput: X.XX Mb/s"
 *
 * ── Example invocation ──────────────────────────────────────────────────────
 *   ./ns3 run "scratch/4.2.6/wifi-mlo-latency-scenario5 \
 *       --numLinks=2 --numObssStas1=4 --numObssStas2=4 \
 *       --offeredLoad=371 --RngRun=1"
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

NS_LOG_COMPONENT_DEFINE("WifiMloLatencyScenario5");

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
//  Main
// ---------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
    // -----------------------------------------------------------------------
    //  Parameters
    // -----------------------------------------------------------------------
    double   simTime        = 10.0;   // s
    uint32_t payloadSize    = 1500;   // bytes
    uint32_t nMpdus         = 1024;   // A-MPDU limit
    double   offeredLoad    = 370.0;  // Mb/s  (main BSS DL)
    uint32_t numLinks       = 2;      // 1 = SLO,  2 = MLO-STR
    uint32_t numObssStas1   = 1;      // OBSS-1 stations on ch0
    uint32_t numObssStas2   = 1;      // OBSS-2 stations on ch1 (MLO only)
    double   obssRatePerSta = 10.0;   // Mb/s per OBSS STA — 10 % of single-link
                                      // 20 MHz / MCS 9 / 2 SS goodput (~100 Mb/s)
                                      // matches Carrascosa TON [17] 20 MHz setup.
    double   startupGuard   = 1.0;    // s idle before traffic starts

    CommandLine cmd(__FILE__);
    cmd.AddValue("simTime",        "Simulation time [s]",                       simTime);
    cmd.AddValue("payloadSize",    "UDP payload size [B]",                       payloadSize);
    cmd.AddValue("nMpdus",         "Max MPDUs per A-MPDU",                      nMpdus);
    cmd.AddValue("offeredLoad",    "Main-BSS DL offered load [Mb/s]",           offeredLoad);
    cmd.AddValue("numLinks",       "Main BSS links: 1=SLO, 2=MLO-STR",         numLinks);
    cmd.AddValue("numObssStas1",   "OBSS-1 STAs on ch0 (primary channel)",     numObssStas1);
    cmd.AddValue("numObssStas2",   "OBSS-2 STAs on ch1 (secondary; MLO only)", numObssStas2);
    cmd.AddValue("obssRatePerSta", "DL rate per OBSS STA [Mb/s]",              obssRatePerSta);
    cmd.AddValue("startupGuard",   "Idle before traffic starts [s]",           startupGuard);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(numLinks != 1 && numLinks != 2,
                    "numLinks must be 1 (SLO) or 2 (MLO-STR)");
    NS_ABORT_MSG_IF(offeredLoad <= 0.0, "offeredLoad must be > 0");

    // For SLO, OBSS-2 (ch1) is irrelevant — force to 0 so no extra nodes are created.
    if (numLinks == 1)
        numObssStas2 = 0;

    RngSeedManager::SetSeed(1); // run index controlled by --RngRun

    std::cout << "=== Latency Scenario #5 ===\n"
              << "numLinks=" << numLinks
              << "  offeredLoad=" << offeredLoad << " Mb/s"
              << "  OBSS1=" << numObssStas1 << " stas"
              << "  OBSS2=" << numObssStas2 << " stas"
              << "  obssRate=" << obssRatePerSta << " Mb/s/sta\n\n";

    // -----------------------------------------------------------------------
    //  Two 5 GHz spectrum channels (80 MHz each)
    //    ch0: centre 42  (~5210 MHz)  → "channel 36"  in thesis (Table 3.12)
    //    ch1: centre 106 (~5530 MHz)  → "channel 100" in thesis
    // -----------------------------------------------------------------------
    struct PhyChan
    {
        uint32_t       chanNum; // ns-3 centre channel number
        FrequencyRange range;
        const char*    band;
    };
    // Carrascosa TON [17] uses 20 MHz channels 36 and 100 (FCB-WACA dataset
    // is 20 MHz only); ns-3 channel numbers are the primary 20 MHz IDs.
    const PhyChan phyChans[2] = {
        {  36, WIFI_SPECTRUM_5_GHZ, "BAND_5GHZ" }, // ch0 — paper's "ch 36"
        { 100, WIFI_SPECTRUM_5_GHZ, "BAND_5GHZ" }, // ch1 — paper's "ch 100"
    };

    // Shared spectrum medium per channel: all nodes on the same medium contend.
    Ptr<MultiModelSpectrumChannel> specMedia[2];
    for (int i = 0; i < 2; ++i)
    {
        specMedia[i] = CreateObject<MultiModelSpectrumChannel>();
        // Residential indoor propagation per thesis Table 3.1 (LogDistance gamma=3.5, refLoss=40 dB @ 1 m).
        Ptr<LogDistancePropagationLossModel> loss = CreateObject<LogDistancePropagationLossModel>();
        loss->SetAttribute("Exponent",          DoubleValue(3.5));
        loss->SetAttribute("ReferenceDistance", DoubleValue(1.0));
        loss->SetAttribute("ReferenceLoss",     DoubleValue(40.0));
        specMedia[i]->AddPropagationLossModel(loss);
        specMedia[i]->SetPropagationDelayModel(
            CreateObject<ConstantSpeedPropagationDelayModel>());
    }

    // -----------------------------------------------------------------------
    //  MAC queue size
    // -----------------------------------------------------------------------
    Config::SetDefault("ns3::WifiMacQueue::MaxSize", QueueSizeValue(QueueSize("4096p")));

    // -----------------------------------------------------------------------
    //  WiFi helpers — EHT 802.11be, MCS 11, 2 SS, GI 800 ns
    //  (Two helpers: one for the main BSS, one for OBSS single-link nodes)
    // -----------------------------------------------------------------------
    // Carrascosa TON [17] §II-A: "we employ a fixed modulation and coding
    // scheme on both interfaces — 256-QAM with coding rate 5/6 and 2 spatial
    // streams" → EHT MCS 9.
    const char* DATA_MODE = "EhtMcs9";
    const char* CTL_MODE  = "EhtMcs0";

    auto configWifi = [&](WifiHelper& wifi, uint32_t links) {
        wifi.SetStandard(WIFI_STANDARD_80211be);
        wifi.ConfigHeOptions("GuardInterval", TimeValue(NanoSeconds(800)));
        for (uint32_t l = 0; l < links; ++l)
            wifi.SetRemoteStationManager(l,
                                         "ns3::ConstantRateWifiManager",
                                         "DataMode",    StringValue(DATA_MODE),
                                         "ControlMode", StringValue(CTL_MODE));
    };

    WifiHelper wifiMain, wifiObss;
    configWifi(wifiMain, numLinks);
    configWifi(wifiObss, 1);

    // -----------------------------------------------------------------------
    //  Main BSS nodes + WiFi installation
    // -----------------------------------------------------------------------
    NodeContainer mainApNode, mainStaNode;
    mainApNode.Create(1);
    mainStaNode.Create(1);

    NetDeviceContainer mainApDev, mainStaDev;
    {
        SpectrumWifiPhyHelper phy(numLinks);
        phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
        if (numLinks > 1)
            phy.SetPcapCaptureType(WifiPhyHelper::PcapCaptureType::PCAP_PER_LINK);

        for (uint32_t l = 0; l < numLinks; ++l)
        {
            std::ostringstream chSet;
            chSet << "{" << phyChans[l].chanNum << ", 20, " << phyChans[l].band << ", 0}";
            phy.AddChannel(specMedia[l], phyChans[l].range);
            phy.Set(l, "ChannelSettings",             StringValue(chSet.str()));
            phy.Set(l, "Antennas",                    UintegerValue(2));
            phy.Set(l, "MaxSupportedTxSpatialStreams", UintegerValue(2));
            phy.Set(l, "MaxSupportedRxSpatialStreams", UintegerValue(2));
        }

        Ssid mainSsid("MAIN");
        WifiMacHelper mac;

        mac.SetType("ns3::ApWifiMac",
                    "Ssid",            SsidValue(mainSsid),
                    "BE_MaxAmpduSize", UintegerValue(nMpdus * (payloadSize + 60)),
                    "MpduBufferSize",  UintegerValue(nMpdus));
        mainApDev = wifiMain.Install(phy, mac, mainApNode);

        mac.SetType("ns3::StaWifiMac",
                    "Ssid",            SsidValue(mainSsid),
                    "ActiveProbing",   BooleanValue(false),
                    "BE_MaxAmpduSize", UintegerValue(nMpdus * (payloadSize + 60)),
                    "MpduBufferSize",  UintegerValue(nMpdus));
        mainStaDev = wifiMain.Install(phy, mac, mainStaNode);
    }

    // -----------------------------------------------------------------------
    //  OBSS groups
    //  Each OBSS occupies one physical channel with a single-link BSS
    //  (1 AP + numObssStasN STAs, each STA receiving DL traffic).
    // -----------------------------------------------------------------------
    struct ObssGroup
    {
        uint32_t    numStas;
        int         chanIdx;
        uint16_t    port; // UdpServer port on OBSS STAs
        NodeContainer  ap;
        NodeContainer  stas;
        NetDeviceContainer apDev;
        NetDeviceContainer staDev;
        std::vector<Ipv4Address> staAddrs;
    };

    // To produce the airtime occupancy the Carrascosa TON [17] paper assumes,
    // each "OBSS station" needs to be an INDEPENDENT BSS (one AP + one STA),
    // not all 7 hanging off the same AP. With N independent OBSS APs all
    // contending via CSMA, the main BSS gets ~1/(N+1) of the airtime — which
    // matches the paper's per-occupancy SLO max much more closely.
    std::vector<ObssGroup> obssGroups;
    for (uint32_t i = 0; i < numObssStas1; ++i)
        obssGroups.push_back({1, 0,
                              static_cast<uint16_t>(5001 + i),
                              {}, {}, {}, {}, {}});
    for (uint32_t i = 0; i < numObssStas2; ++i)
        obssGroups.push_back({1, 1,
                              static_cast<uint16_t>(5100 + i),
                              {}, {}, {}, {}, {}});

    uint32_t obssIdx = 0;
    for (auto& og : obssGroups)
    {
        og.ap.Create(1);
        og.stas.Create(og.numStas);

        SpectrumWifiPhyHelper phy(1);
        phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);

        int ci = og.chanIdx;
        std::ostringstream chSet;
        chSet << "{" << phyChans[ci].chanNum << ", 20, " << phyChans[ci].band << ", 0}";
        phy.AddChannel(specMedia[ci], phyChans[ci].range);
        phy.Set(0, "ChannelSettings",             StringValue(chSet.str()));
        phy.Set(0, "Antennas",                    UintegerValue(2));
        phy.Set(0, "MaxSupportedTxSpatialStreams", UintegerValue(2));
        phy.Set(0, "MaxSupportedRxSpatialStreams", UintegerValue(2));

        std::ostringstream ssidStr;
        ssidStr << "OBSS" << obssIdx++;
        Ssid ssid(ssidStr.str());

        WifiMacHelper mac;
        mac.SetType("ns3::ApWifiMac",
                    "Ssid",            SsidValue(ssid),
                    "BE_MaxAmpduSize", UintegerValue(nMpdus * (payloadSize + 60)),
                    "MpduBufferSize",  UintegerValue(nMpdus));
        og.apDev = wifiObss.Install(phy, mac, og.ap);

        mac.SetType("ns3::StaWifiMac",
                    "Ssid",            SsidValue(ssid),
                    "ActiveProbing",   BooleanValue(false),
                    "BE_MaxAmpduSize", UintegerValue(nMpdus * (payloadSize + 60)),
                    "MpduBufferSize",  UintegerValue(nMpdus));
        og.staDev = wifiObss.Install(phy, mac, og.stas);
    }

    // -----------------------------------------------------------------------
    //  Mobility: all nodes co-located for mutual interference.
    //  AP at origin (0,0,0), STA 5 m away on x-axis.
    // -----------------------------------------------------------------------
    MobilityHelper mob;
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    auto placeAt = [&](NodeContainer& nodes, Vector pos) {
        Ptr<ListPositionAllocator> alloc = CreateObject<ListPositionAllocator>();
        for (uint32_t i = 0; i < nodes.GetN(); ++i)
            alloc->Add(pos);
        mob.SetPositionAllocator(alloc);
        mob.Install(nodes);
    };

    placeAt(mainApNode,  Vector(0.0, 0.0, 0.0));
    placeAt(mainStaNode, Vector(5.0, 0.0, 0.0));
    for (auto& og : obssGroups)
    {
        placeAt(og.ap,   Vector(0.0, 0.0, 0.0));
        placeAt(og.stas, Vector(5.0, 0.0, 0.0));
    }

    // -----------------------------------------------------------------------
    //  Internet stack
    // -----------------------------------------------------------------------
    InternetStackHelper stack;
    stack.Install(mainApNode);
    stack.Install(mainStaNode);
    for (auto& og : obssGroups)
    {
        stack.Install(og.ap);
        stack.Install(og.stas);
    }

    // -----------------------------------------------------------------------
    //  IP addressing
    //  Main BSS : 10.1.0.0/24
    //  OBSS-1   : 10.2.0.0/24
    //  OBSS-2   : 10.3.0.0/24
    // -----------------------------------------------------------------------
    Ipv4AddressHelper addr;

    addr.SetBase("10.1.0.0", "255.255.255.0");
    addr.Assign(mainApDev);
    Ipv4InterfaceContainer mainStaIf = addr.Assign(mainStaDev);
    Ipv4Address mainStaAddr = mainStaIf.GetAddress(0);

    addr.NewNetwork();
    for (auto& og : obssGroups)
    {
        addr.Assign(og.apDev);
        Ipv4InterfaceContainer staIf = addr.Assign(og.staDev);
        for (uint32_t s = 0; s < og.numStas; ++s)
            og.staAddrs.push_back(staIf.GetAddress(s));
        addr.NewNetwork();
    }

    // -----------------------------------------------------------------------
    //  Traffic
    // -----------------------------------------------------------------------
    const double tStart        = startupGuard;
    const double tStop         = simTime + startupGuard;
    const double burstDataRate = 5.0e9;           // >> all offered loads → one pkt per on-period
    const double pktBits       = payloadSize * 8.0;
    const double pktOnSec      = pktBits / burstDataRate; // ~2.4 µs

    std::ostringstream drStr;
    drStr << static_cast<uint64_t>(burstDataRate) << "bps";

    // ── Main BSS DL (AP → STA, port 5000) ──────────────────────────────────
    {
        const double iatSec    = pktBits / (offeredLoad * 1e6);
        const double offMean   = iatSec - pktOnSec;
        NS_ABORT_MSG_IF(offMean <= 0.0,
                        "offeredLoad exceeds burst data rate — reduce load");

        std::ostringstream onStr, offStr;
        onStr  << "ns3::ConstantRandomVariable[Constant=" << pktOnSec  << "]";
        offStr << "ns3::ExponentialRandomVariable[Mean="  << offMean   << "]";

        OnOffHelper onoff("ns3::UdpSocketFactory",
                          InetSocketAddress(mainStaAddr, 5000));
        onoff.SetAttribute("DataRate",   StringValue(drStr.str()));
        onoff.SetAttribute("PacketSize", UintegerValue(payloadSize));
        onoff.SetAttribute("OnTime",     StringValue(onStr.str()));
        onoff.SetAttribute("OffTime",    StringValue(offStr.str()));

        ApplicationContainer apps = onoff.Install(mainApNode);
        apps.Start(Seconds(tStart));
        apps.Stop(Seconds(tStop));

        UdpServerHelper srv(5000);
        ApplicationContainer srvApps = srv.Install(mainStaNode);
        srvApps.Start(Seconds(tStart));
        srvApps.Stop(Seconds(tStop + 1.0));
    }

    // ── OBSS DL (each OBSS AP → its STAs, port 5001/5002) ──────────────────
    for (auto& og : obssGroups)
    {
        const double obssIat  = pktBits / (obssRatePerSta * 1e6);
        const double obssOff  = obssIat - pktOnSec;
        NS_ABORT_MSG_IF(obssOff <= 0.0,
                        "obssRatePerSta exceeds burst data rate");

        std::ostringstream onStr, offStr;
        onStr  << "ns3::ConstantRandomVariable[Constant=" << pktOnSec << "]";
        offStr << "ns3::ExponentialRandomVariable[Mean="  << obssOff  << "]";

        // One UdpServer per OBSS STA
        UdpServerHelper srv(og.port);
        ApplicationContainer srvApps = srv.Install(og.stas);
        srvApps.Start(Seconds(tStart));
        srvApps.Stop(Seconds(tStop + 1.0));

        // One OnOff flow from the OBSS AP to each of its STAs
        for (uint32_t s = 0; s < og.numStas; ++s)
        {
            OnOffHelper onoff("ns3::UdpSocketFactory",
                              InetSocketAddress(og.staAddrs[s], og.port));
            onoff.SetAttribute("DataRate",   StringValue(drStr.str()));
            onoff.SetAttribute("PacketSize", UintegerValue(payloadSize));
            onoff.SetAttribute("OnTime",     StringValue(onStr.str()));
            onoff.SetAttribute("OffTime",    StringValue(offStr.str()));

            ApplicationContainer apps = onoff.Install(og.ap);
            apps.Start(Seconds(tStart));
            apps.Stop(Seconds(tStop));
        }
    }

    // -----------------------------------------------------------------------
    //  Channel-access-delay traces (main AP only = NodeList/0)
    //  + MacRx on main STA (NodeList/1)
    // -----------------------------------------------------------------------
    Config::Connect("/NodeList/0/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx",
                    MakeCallback(&ChannelAccessMacTxTrace));
    Config::Connect("/NodeList/0/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyTxBegin",
                    MakeCallback(&ChannelAccessPhyTxBeginTrace));
    Config::ConnectWithoutContext(
        "/NodeList/1/DeviceList/*/$ns3::WifiNetDevice/Mac/MacRx",
        MakeCallback(&MacRxTrace));

    // -----------------------------------------------------------------------
    //  Run
    // -----------------------------------------------------------------------
    Simulator::Stop(Seconds(tStop + 2.0));
    Simulator::Run();

    // -----------------------------------------------------------------------
    //  Results (main BSS only)
    // -----------------------------------------------------------------------
    const double meanLatMs = ChannelAccessMeanMs();
    const double p50Ms     = ChannelAccessPercentileMs(50.0);
    const double p95Ms     = ChannelAccessPercentileMs(95.0);
    const double p99Ms     = ChannelAccessPercentileMs(99.0);
    const double meanThrMbps =
        (simTime > 0.0) ? g_rxBytes * 8.0 / (simTime * 1e6) : 0.0;

    std::cout << "\nMean DL Latency: "   << meanLatMs   << " ms\n";
    std::cout << "DL Latency p50: "     << p50Ms       << " ms\n";
    std::cout << "DL Latency p95: "     << p95Ms       << " ms\n";
    std::cout << "DL Latency p99: "     << p99Ms       << " ms\n";
    std::cout << "Mean DL Throughput: " << meanThrMbps << " Mb/s\n";

    Simulator::Destroy();
    return 0;
}
