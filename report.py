"""
Generate a PDF file report containing information about the computer hardware.

Check those libraries for additional information:
    https://docs.python.org/3/library/platform.html#platform.system
    https://psutil.readthedocs.io/en/latest/
    https://docs.python.org/3/library/socket.html
"""
import datetime as dt
import os
import platform
import socket
from collections import defaultdict
from glob import glob

import pandas as pd
import psutil
import utils
from psutil._common import bytes2human

PdfBuilder = utils.PdfBuilder()
GraphBuilder = utils.GraphBuilder()


# The default analysis period will have start_time as the current time and
# stop_time as the current time plus 1 minute.
# It will therefore be a one-minute report. For testing, just run the code.

current_time = dt.datetime.now()
start_time = current_time
stop_time = start_time + dt.timedelta(minutes=1)


class HardwareReport:
    """Generate a PDF file with the computer hardware report.

    Contain numerical and graphical information of the System, CPU,
    Memory, Disks and Network.
    """

    def __init__(
        self,
        start_time: dt,
        stop_time: dt,
        system: bool = True,
        cpu: bool = True,
        memory: bool = True,
        disk: bool = True,
        network: bool = True,
    ) -> PdfBuilder:
        """
        Contain the initial arguments.

        :args
            start_time: <datetime> Start time for analysis
            stop_time: <datetime> Stop time for analysis
            system = <bool> Set the system for analysis
            memory = <bool> Set the memory for analysis
            cpu = <bool Set the cpu for analysis
            disk = <bool> Set the disk for analysis
            network = <bool> Set the network for analysis

        :output
            pdf file with the computer hardware report
        """
        self.start_time = start_time
        self.stop_time = stop_time
        self.system = system
        self.cpu = cpu
        self.memory = memory
        self.disk = disk
        self.network = network

        current_time = dt.datetime.now()
        # Current formatted time is used as part of the PDF file name
        current_formatted_time = current_time.strftime('%d_%m_%Y_%H_%M_%S')

        # A new PDF file will be created in each run of the code
        # e.g: Harware_Report_01_08_2022_11_19_56.pdf
        self.canvas = PdfBuilder.create_template(
            current_formatted_time, PdfBuilder.pagesize
        )

        # This variable "parts" will append all parts (text, images, tables)
        # for the PDF
        self.parts = []
        self.parts.append(PdfBuilder.logo)

        # Header
        text = f"""
        COMPUTER HARDWARE REPORT
        Date: {current_time.strftime("%d/%m/%Y")}
        Time: {current_time.strftime("%H:%M:%S")}

        """
        self.parts.append(PdfBuilder.format_text(text))

        # CPU and memory have data that vary with the analyzed time and
        # therefore need to have their values accumulated for the final values.
        if cpu:
            self.cpu_data = defaultdict(list)  # Accumulate CPU data
        if memory:
            # Return statistics about system memory usage
            self.svmem = psutil.virtual_memory()
            # Return system swap memory statistics
            self.swap = psutil.swap_memory()
            self.memory_data = defaultdict(list)  # Accumulate memory data
            self.swap_data = defaultdict(list)  # Accumulate swap data

        # Set a new current time to be trusted with the parse time
        current_time = dt.datetime.now()
        while start_time.timestamp() < current_time.timestamp() < stop_time.timestamp():

            if cpu:
                # If supported return a list of frequencies for each CPU
                for i, percentage in enumerate(
                    psutil.cpu_percent(percpu=True, interval=1)
                ):
                    self.cpu_data[f'Core_{i}'].append(percentage)

                # Append a float with the current system-wide CPU utilization
                # as a percentage
                self.cpu_data['Total'].append(psutil.cpu_percent())
                # Append the current datetime
                self.cpu_data['Datetime'].append(current_time)

            if memory:
                # Append the total physical memory in bytes (exclusive swap).
                self.memory_data['Total'].append(
                    float(
                        self.convert_bytes_to_readable_measurement(self.svmem.total)[
                            :-2
                        ]
                    )
                )
                # Append the total swap memory in bytes
                self.swap_data['Total'].append(
                    float(
                        self.convert_bytes_to_readable_measurement(self.swap.total)[
                            :-2]
                    )
                )
                # Append the memory (in bytes) to processes without the system
                # going into swap
                self.memory_data['Available'] = float(
                    self.convert_bytes_to_readable_measurement(self.svmem.available)[
                        :-2
                    ]
                )
                # Append free swap memory in bytes
                # Free: Memory not being used at all that is readily available;
                # note that this doesn’t reflect the actual memory available
                self.swap_data['Free'] = float(
                    self.convert_bytes_to_readable_measurement(self.swap.free)[
                        :-2]
                )
                # Append memory used, calculated differently depending on the
                # platform and designed for informational purposes only
                self.memory_data['Used'].append(
                    float(
                        self.convert_bytes_to_readable_measurement(self.svmem.used)[
                            :-2]
                    )
                )
                # Append used swap memory in bytes
                self.swap_data['Used'].append(
                    float(
                        self.convert_bytes_to_readable_measurement(self.swap.used)[
                            :-2]
                    )
                )

                # Append the percentage usage calculated as (total - available)
                # / total * 100
                self.memory_data['Percentage_usage'].append(self.svmem.percent)
                # Append the percentage usage calculated as (total - free) /
                # total * 100
                self.swap_data['Percentage_usage'].append(self.swap.percent)
                # Append the current datetime
                self.memory_data['Datetime'].append(current_time)
                # Append the current datetime
                self.swap_data['Datetime'].append(current_time)

            current_time = dt.datetime.now()  # Updates the current time

    def convert_bytes_to_readable_measurement(self, byte_value, suffix='B'):
        """
        Scale bytes to its proper format.

        e.g:
            1253656 => '1.20MB'
            1253656678 => '1.17GB'
        """
        factor = 1024
        for unit in ['', 'K', 'M', 'G', 'T', 'P']:
            if byte_value < factor:
                return f'{byte_value:.2f}{unit}{suffix}'
            byte_value /= factor

    def generate_pdf(self):
        """Contain all functions to generate the PDF report."""
        if self.system:
            """
            System Information with six attributes:
                System: Return the system/OS name, such as 'Linux', 'Darwin', 'Windows'.
                Node Name: Return the network name (may not be fully qualified!).
                Release: Return the system’s release, e.g. '2.2.0' or 'NT'.
                Version: Return the system’s release version, e.g. '#3 on degas'.
                Machine: Return the machine type, e.g. 'AMD64'.
                Processor: Return the (real) processor name, e.g. 'amdk6'.
                Boot Time: Return the system boot time in seconds since the epoch.
                    On Windows this function may return a time which is off by 1 second
                    if it’s used across different processes.
            """
            # Return a tuple() containing the six system attributes
            uname = platform.uname()

            boot_time_timestamp = psutil.boot_time()
            bt = dt.datetime.fromtimestamp(boot_time_timestamp)

            title = ' System Information '
            text = f"""{title.center(91, "=")}\n\n
            System: {uname.system}
            Node Name: {uname.node}
            Release: {uname.release}
            Version: {uname.version}
            Machine: {uname.machine}
            Processor: {uname.processor}
            Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}

            """
            self.parts.append(PdfBuilder.format_text(text))

        if self.cpu:
            """
            CPU information with 5 attributes:
                Physical cores: The number of physical cores
                Total cores: The number of logical CPUs in the system
                Max Frequency: Maximum CPU frequency
                Min Frequency: Minimum CPU frequency
                Current Frequency: Current CPU frequency
            """
            title = ' CPU Info '
            # CPU frequencies in Mhz (current, min and max)
            # On Linux current frequency reports the real-time value,
            # on all others it usually represents the nominal “fixed” value.
            cpufreq = psutil.cpu_freq()

            text = f"""{title.center(89, "=")}\n\n
            Physical cores: {psutil.cpu_count(logical=False)}
            Total cores: {psutil.cpu_count(logical=True)}
            Max Frequency: {cpufreq.max:.2f}Mhz
            Min Frequency: {cpufreq.min:.2f}Mhz
            Current Frequency: {cpufreq.current:.2f}Mhz
            """
            cpu_plot_filename = 'cpu_plot.png'
            # Sets the CPU data into a Dataframe
            core = pd.DataFrame(self.cpu_data)
            # Sets the datetime as the new index
            core.set_index('Datetime', inplace=True)
            # Creates the CPU plot
            GraphBuilder.lineplot('CPU', core, cpu_plot_filename)

            text += 'Peak CPU Usage:'
            for i in range(len(core.max())):
                text += f"""
                \t{core.columns[i]}: {core.max()[i]}%"""

            self.parts.append(PdfBuilder.format_text(text))
            self.parts.append(PdfBuilder.format_image(
                filename=cpu_plot_filename))
            self.parts.append(PdfBuilder.go_next_page())

        if self.memory:
            """
            Memory/Swap Information with 4 attributes:

            Memory:
                Total: Total physical memory (exclusive swap).
                Available: Memory that can be given instantly to
                        processes without the system going into swap
                Used: Memory used, calculated differently depending on the
                    platform and designed for informational purposes only
                Used Percentage: The percentage usage calculated as:
                                    (total - available) / total * 100
            Swap:
                Total: Total swap memory
                Free: Free swap memory
                Used: Used swap memory
                Percentage usage: The percentage usage calculated as:
                                (total - free) / total * 100

            Ps. Swap memory doesn't have the attribute "available",
            we use "free".
            """
            title1 = ' Memory Information '
            text = f"""{title1.center(91, "=")}\n\n
            Total: {self.convert_bytes_to_readable_measurement(
                self.svmem.total)}
            Available: {self.convert_bytes_to_readable_measurement(
                self.svmem.available)}
            Used: {self.convert_bytes_to_readable_measurement(self.svmem.used)}
            Percentage usage: {self.svmem.percent}%
            """
            self.parts.append(PdfBuilder.format_text(text))

            # Sets the memory data into a Dataframe
            memory = pd.DataFrame(self.memory_data)
            # Sets the datetime as the new index
            memory.set_index('Datetime', inplace=True)
            # Creates the memory plot
            plot_filename = 'memory_plot.png'
            GraphBuilder.lineplot('Memory', memory, plot_filename)
            self.parts.append(PdfBuilder.format_image(filename=plot_filename))
            self.parts.append(PdfBuilder.go_next_page())

            title2 = ' SWAP '
            text = f"""{title2.center(87, "=")}\n\n
            Total: {self.convert_bytes_to_readable_measurement(
                self.swap.total)}
            Free: {self.convert_bytes_to_readable_measurement(self.swap.free)}
            Used: {self.convert_bytes_to_readable_measurement(self.swap.used)}
            Percentage usage: {self.swap.percent}%
            """
            self.parts.append(PdfBuilder.format_text(text))

            # Sets the swap data into a Dataframe
            memory = pd.DataFrame(self.swap_data)
            # Sets the datetime as the new index
            memory.set_index('Datetime', inplace=True)
            # Creates the swap plot
            plot_filename = 'swap_plot.png'
            GraphBuilder.lineplot('Swap', memory, plot_filename)
            self.parts.append(PdfBuilder.format_image(filename=plot_filename))

        if self.disk:
            """
            Disk Information with 6 attributes:
                Device: The device path (e.g. "/dev/hda1").
                    On Windows this is the drive letter (e.g. "C:\\").
                Mount point: The mount point path (e.g. "/").
                    On Windows this is the drive letter (e.g. "C:\\").
                Filesystem type: The partition filesystem (e.g. "ext3" on UNIX
                    or "NTFS" on Windows).
                Total: Total space expressed in bytes.
                Free: Free space expressed in bytes.
                Percentage usage: Used sapce expressed in Percentage.
            """
            # Return all mounted disk partitions as a list of named tuples
            # including device, mount point and filesystem type, similarly
            # to “df” command on UNIX
            partitions = psutil.disk_partitions()
            title = ' Disk Information '
            text = f"""{title.center(92, "=")}\n\n
            """
            self.parts.append(PdfBuilder.format_text(text))

            # Sets the table header for the Disks information
            data = [['Device', 'Mount', 'Fstype',
                     'Total', 'Used', 'Free', 'Usage(%)']]
            for i, partition in enumerate(partitions):
                try:
                    partition_usage = psutil.disk_usage(partition.mountpoint)
                    # Creates a table with the disk information
                    data.append(
                        [
                            f'{partition.device}',
                            f'{partition.mountpoint}',
                            f'{partition.fstype}',
                            f'{self.convert_bytes_to_readable_measurement(partition_usage.total)}',
                            f'{self.convert_bytes_to_readable_measurement(partition_usage.used)}',
                            f'{self.convert_bytes_to_readable_measurement(partition_usage.free)}',
                            f'{partition_usage.percent}%',
                        ]
                    )
                except PermissionError:
                    # This can be catched due to the disk that isn't ready
                    continue
            self.parts.append(PdfBuilder.format_table(data))

            # Return IO statistics since boot
            disk_io = psutil.disk_io_counters()

            text = f"""
            Total bytes read: {self.convert_bytes_to_readable_measurement(disk_io.read_bytes)}
            Total bytes written: {self.convert_bytes_to_readable_measurement(disk_io.write_bytes)}
            """
            self.parts.append(PdfBuilder.format_text(text))
            self.parts.append(PdfBuilder.go_next_page())

        if self.network:
            """
            Network Information:
            System-wide network I/O statistics as a named tuple including
            many attributes. A clone of 'ifconfig' on UNIX.

            These part of the code below is based
            on the Giampaolo Rodola's code.
            Find it here:
            https://github.com/giampaolo/psutil/blob/master/scripts/ifconfig.py

            lo:
                stats          : speed=0MB, duplex=?, mtu=65536, up=yes
                incoming       : bytes=1.95M, pkts=22158, errs=0, drops=0
                outgoing       : bytes=1.95M, pkts=22158, errs=0, drops=0
                IPv4 address   : 127.0.0.1
                    netmask   : 255.0.0.0
                IPv6 address   : ::1
                    netmask   : ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff
                MAC  address   : 00:00:00:00:00:00
            docker0:
                stats          : speed=0MB, duplex=?, mtu=1500, up=yes
                incoming       : bytes=3.48M, pkts=65470, errs=0, drops=0
                outgoing       : bytes=164.06M, pkts=112993, errs=0, drops=0
                IPv4 address   : 172.17.0.1
                    broadcast : 172.17.0.1
                    netmask   : 255.255.0.0
                IPv6 address   : fe80::42:27ff:fe5e:799e%docker0
                    netmask   : ffff:ffff:ffff:ffff::
                MAC  address   : 02:42:27:5e:79:9e
                    broadcast : ff:ff:ff:ff:ff:ff
            wlp3s0:
                stats          : speed=0MB, duplex=?, mtu=1500, up=yes
                incoming       : bytes=7.04G, pkts=5637208, errs=0, drops=0
                outgoing       : bytes=372.01M, pkts=3200026, errs=0, drops=0
                IPv4 address   : 10.0.0.2
                    broadcast : 10.255.255.255
                    netmask   : 255.0.0.0
                IPv6 address   : fe80::ecb3:1584:5d17:937%wlp3s0
                    netmask   : ffff:ffff:ffff:ffff::
                MAC  address   : 48:45:20:59:a4:0c
                    broadcast : ff:ff:ff:ff:ff:ff
            """
            title = ' Network Information '
            text = f"{title.center(92, '=')}\n\n"
            self.parts.append(PdfBuilder.format_text(text))

            # The address (and protocol) families
            af_map = {
                socket.AF_INET: 'IPv4',
                socket.AF_INET6: 'IPv6',
                psutil.AF_LINK: 'MAC',
            }
            # The duplex communication type; it can be either
            # NIC_DUPLEX_FULL, NIC_DUPLEX_HALF or NIC_DUPLEX_UNKNOWN.
            duplex_map = {
                psutil.NIC_DUPLEX_FULL: 'full',
                psutil.NIC_DUPLEX_HALF: 'half',
                psutil.NIC_DUPLEX_UNKNOWN: '?',
            }

            # Return information about each NIC (network interface card)
            # installed on the system
            stats = psutil.net_if_stats()
            # Return system-wide network I/O statistics as a named tuple
            io_counters = psutil.net_io_counters(pernic=True)

            # Creates a table with the network information (with no grid).
            data = []
            for nic, addrs in psutil.net_if_addrs().items():
                data.append([f'{nic}:', '', ''])
                if nic in stats:
                    st = stats[nic]
                    data.append(
                        [
                            '',
                            'stats',
                            f": speed={st.speed}MB, duplex={duplex_map[st.duplex]}, mtu={st.mtu}, up={'yes' if st.isup else 'no'}",
                        ]
                    )
                if nic in io_counters:
                    io = io_counters[nic]
                    data.append(
                        [
                            '',
                            'incoming',
                            f': bytes={bytes2human(io.bytes_recv)}, pkts={io.packets_recv}, errs={io.errin}, drops={io.dropin}',
                        ]
                    )
                    data.append(
                        [
                            '',
                            'outgoing',
                            f': bytes={bytes2human(io.bytes_sent)}, pkts={io.packets_sent}, errs={io.errout}, drops={io.dropout}',
                        ]
                    )
                for addr in addrs:
                    data.append(
                        [
                            '',
                            f'{af_map.get(addr.family, addr.family):4} address',
                            f': {addr.address}',
                        ]
                    )
                    if addr.broadcast:
                        data.append(
                            ['', '        broadcast', f': {addr.broadcast}'])
                    if addr.netmask:
                        data.append(
                            ['', '        netmask', f': {addr.netmask}'])
                    if addr.ptp:
                        data.append(['', '        p2p', f': {addr.ptp}'])
            self.parts.append(PdfBuilder.unformatted_table(data))

        # Creates the final PDF file with all the appended parts
        PdfBuilder.build_pdf(self.canvas, self.parts)

        # Delete all saved graph files
        plots = glob('*.png')
        for plot in plots:
            os.remove(plot)


if __name__ == '__main__':
    report = HardwareReport(
        start_time=start_time,
        stop_time=stop_time,
        system=True,
        cpu=True,
        memory=True,
        disk=True,
        network=True,
    )
    report.generate_pdf()
