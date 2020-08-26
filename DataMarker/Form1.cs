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
        public int current_index;
        public List<FileInfo> file_list;
        public Dictionary<string,bool> recorded_labels;

        public Form1()
        {
            InitializeComponent();
            this.data_path = "";
        }
        private void init_status()
        {
            this.recorded_labels = new Dictionary<string, bool>();
            this.list_path = this.data_path + "\\valid_file.txt";
            FileStream FS = new FileStream(this.list_path, FileMode.OpenOrCreate);
            StreamReader SR = new StreamReader(FS);
            while (!SR.EndOfStream)
            {
                string[] content = SR.ReadLine().Split('\t');
                if (content.Length == 2)
                {
                    this.recorded_labels.Add(content[0], content[1].Equals("1"));
                }
            }
            DirectoryInfo DI = new DirectoryInfo(this.data_path);
            this.file_list = DI.GetFiles().ToList<FileInfo>();
            this.file_list.Remove(this.file_list[this.file_list.Count - 1]);
            this.current_index = 0;
            this.ImageBox.Load(this.file_list[this.current_index].FullName);
            this.refresh_label();
        }

        private void refresh_label()
        {
            string current_img = this.file_list[this.current_index].Name;
            if (this.recorded_labels.ContainsKey(current_img))
            {
                this.mark_pass.Enabled = !this.recorded_labels[current_img];
                this.mark_fail.Enabled = this.recorded_labels[current_img];
            }
            else
            {
                this.mark_pass.Enabled = true;
                this.mark_fail.Enabled = true;
            }
        }

        private void load_list_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog OFD = new FolderBrowserDialog();
            OFD.Description = "选择数据及列表所在目录";
            if (OFD.ShowDialog() == DialogResult.OK)
            {
                this.data_path = OFD.SelectedPath;
                this.init_status();
            }
        }

        private void prev_img_Click(object sender, EventArgs e)
        {
            this.current_index--;
            if (this.current_index < 0)
            {
                this.current_index = 0;
            }
            this.ImageBox.Load(this.file_list[this.current_index].FullName);
            this.refresh_label();
        }

        private void next_img_Click(object sender, EventArgs e)
        {
            this.current_index++;
            if (this.current_index >= this.file_list.Count)
            {
                this.current_index = this.file_list.Count - 1;
            }
            this.ImageBox.Load(this.file_list[this.current_index].FullName);
            this.refresh_label();
        }

        private void mark_pass_Click(object sender, EventArgs e)
        {
            string current_file = this.file_list[this.current_index].Name;
            if (this.recorded_labels.ContainsKey(current_file))
            {
                recorded_labels[current_file] = true;
            }
            else
            {
                recorded_labels.Add(current_file, true);
            }
            this.refresh_label();
        }

        private void mark_fail_Click(object sender, EventArgs e)
        {
            string current_file = this.file_list[this.current_index].Name;
            if (this.recorded_labels.ContainsKey(current_file))
            {
                recorded_labels[current_file] = false;
            }
            else
            {
                recorded_labels.Add(current_file, false);
            }
            this.refresh_label();
        }
    }
}
