using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DataMarker
{
    public partial class Form1 : Form
    {
        public string data_path;
        public string list_path;
        public List<FileInfo> file_list;
        public Form1()
        {
            InitializeComponent();
            this.data_path = "";
        }
        private void init_status()
        {
            this.list_path = this.data_path + "\\valid_file.txt";
            DirectoryInfo DI = new DirectoryInfo(this.data_path);
            this.file_list = DI.GetFiles().ToList<FileInfo>();
            this.file_list.Remove(this.file_list[this.file_list.Count - 1]);

        }
        private void load_list_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog OFD = new FolderBrowserDialog();
            OFD.Description = "选择数据及列表所在目录";
            if (OFD.ShowDialog() == DialogResult.OK)
            {
                this.data_path = OFD.SelectedPath;
            }
            this.init_status();
        }
    }
}
