// WiiBoardBridge.cs
// -----------------------------------------------------------------------------
// Thin bridge: reuse the proven Windows Bluetooth stack (32Feet.NET + WiimoteLib,
// the same combo WiiBalanceWalker uses) and forward per-corner kilograms to the
// Python WELAB ingest (wbb_bridge.py) over localhost UDP.
//
// Wire format, one datagram per WiimoteChanged event (ASCII):
//      <t_seconds>,<tr_kg>,<tl_kg>,<br_kg>,<bl_kg>
//
// BUILD (on the target Windows machine):
//   - Reference WiimoteLib.dll  (use lshachar's or BrianPeek's build)
//   - 32Feet.NET (InTheHand.Net.*) is pulled in transitively by WiimoteLib for BT
//   - Target .NET Framework 4.x (matches the WiimoteLib builds in the wild)
//   - Pair the board ONCE via Windows Bluetooth Settings (not Control Panel);
//     use the permanent-PIN trick so it reconnects on the front button.
//
// NOTE: not compiled/validated here (no hardware/.NET in this sandbox).
//       Verify property names against YOUR WiimoteLib build — the public ones
//       (BalanceBoardState.SensorValuesKg.{TopRight,TopLeft,BottomRight,BottomLeft},
//        WeightKg, ExtensionType.BalanceBoard) are stable across the common forks.
// -----------------------------------------------------------------------------

using System;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using WiimoteLib;

class WiiBoardBridge
{
    const string Host = "127.0.0.1";
    const int Port = 8674;

    static readonly UdpClient Udp = new UdpClient();
    static readonly Stopwatch Clock = new Stopwatch();
    static readonly IPEndPoint EndPoint = new IPEndPoint(IPAddress.Parse(Host), Port);

    static void Main()
    {
        var wm = new Wiimote();
        wm.WiimoteChanged += OnWiimoteChanged;

        Console.WriteLine("Connecting to Balance Board... (press the red sync / front button)");
        try
        {
            wm.Connect();
        }
        catch (Exception ex)
        {
            Console.WriteLine("Connect failed: " + ex.Message);
            Console.WriteLine("Make sure the board is paired in Windows Bluetooth Settings.");
            return;
        }

        if (wm.WiimoteState.ExtensionType != ExtensionType.BalanceBoard)
        {
            Console.WriteLine("Connected device is not a Balance Board. Aborting.");
            wm.Disconnect();
            return;
        }

        // Light an LED so the user knows it's live; start the clock.
        try { wm.SetLEDs(true, false, false, false); } catch { /* non-fatal */ }
        Clock.Start();

        Console.WriteLine("Streaming corner kg to udp://" + Host + ":" + Port +
                          ". Press Enter to stop.");
        Console.ReadLine();

        Clock.Stop();
        wm.WiimoteChanged -= OnWiimoteChanged;
        wm.Disconnect();
        Udp.Close();
    }

    static void OnWiimoteChanged(object sender, WiimoteChangedEventArgs e)
    {
        BalanceBoardState bb = e.WiimoteState.BalanceBoardState;

        // WiimoteLib already calibrated each corner to kg using the board's
        // stored Kg0/Kg17/Kg34 anchors (same piecewise interp as wbb_core).
        double t = Clock.Elapsed.TotalSeconds;
        float tr = bb.SensorValuesKg.TopRight;
        float tl = bb.SensorValuesKg.TopLeft;
        float br = bb.SensorValuesKg.BottomRight;
        float bl = bb.SensorValuesKg.BottomLeft;

        // Invariant culture so the decimal separator is always '.' for the parser.
        string line = string.Format(
            System.Globalization.CultureInfo.InvariantCulture,
            "{0:F4},{1:F3},{2:F3},{3:F3},{4:F3}", t, tr, tl, br, bl);

        byte[] payload = Encoding.ASCII.GetBytes(line);
        try { Udp.Send(payload, payload.Length, EndPoint); }
        catch { /* drop on transient socket error; acquisition must not crash */ }
    }
}
